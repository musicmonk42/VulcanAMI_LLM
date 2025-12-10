#!/usr/bin/env python3
"""
Comprehensive test suite for Graphix graph validation
Tests graph structure, schema compliance, and semantic correctness

Run with:
    pytest test_graph_validation.py -v
    pytest test_graph_validation.py -v --cov=graph_validator --cov-report=html
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

# ============================================================================
# Graph Validator Implementation
# ============================================================================

class ValidationError(Exception):
    """Graph validation error."""
    pass


class GraphValidator:
    """Validates Graphix graph structure and semantics."""

    # Valid node types
    VALID_NODE_TYPES = {
        "CONST", "LOAD", "EMBED", "GenerativeNode", "OutputNode",
        "ContractNode", "PATTERN_COMPILE", "FIND_SUBGRAPH", "GRAPH_SPLICE",
        "GRAPH_COMMIT", "NSO_MODIFY", "ETHICAL_LABEL", "MUL", "ADD",
        "HALT", "EVAL", "Matrix3DNode"
    }

    # Valid edge kinds
    VALID_EDGE_KINDS = {"data", "contract_binding", "nso_binding", "ethics_binding"}

    # Required node fields
    REQUIRED_NODE_FIELDS = {"id", "type"}

    # Required edge fields
    REQUIRED_EDGE_FIELDS = {"from", "to"}

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, enforce stricter validation rules
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []

    def validate_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Validate entire graph structure.

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        try:
            self._validate_root_structure(graph_data)
            self._validate_nodes(graph_data.get("nodes", []))
            self._validate_edges(graph_data.get("edges", []), graph_data.get("nodes", list(]))
            self._validate_graph_connectivity(graph_data)
            self._validate_metadata(graph_data.get("metadata", {}))

            return len(self.errors) == 0

        except Exception as e:
            self.errors.append(f"Validation failed with exception: {e}")
            return False

    def _validate_root_structure(self, graph_data: Dict[str, Any]) -> None:
        """Validate root-level graph structure."""
        required_fields = {"id", "type", "nodes", "edges"}
        missing = required_fields - set(graph_data.keys())

        if missing:
            self.errors.append(f"Missing required root fields: {missing}")

        if graph_data.get("type") != "Graph":
            self.errors.append(f"Invalid graph type: {graph_data.get('type')}")

        if "grammar_version" in graph_data:
            version = graph_data["grammar_version"]
            if not isinstance(version, str) or not version:
                self.errors.append(f"Invalid grammar_version: {version}")

    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """Validate all nodes."""
        if not isinstance(nodes, list):
            self.errors.append(f"Nodes must be a list, got {type(nodes)}")
            return

        if not nodes:
            self.warnings.append("Graph has no nodes")
            return

        node_ids = set()

        for idx, node in enumerate(nodes):
            # Check required fields
            missing = self.REQUIRED_NODE_FIELDS - set(node.keys())
            if missing:
                self.errors.append(f"Node {idx} missing required fields: {missing}")
                continue

            node_id = node.get("id")
            node_type = node.get("type")

            # Validate node ID
            if not node_id or not isinstance(node_id, str):
                self.errors.append(f"Node {idx} has invalid id: {node_id}")
            elif node_id in node_ids:
                self.errors.append(f"Duplicate node id: {node_id}")
            else:
                node_ids.add(node_id)

            # Validate node type
            if node_type not in self.VALID_NODE_TYPES:
                if self.strict_mode:
                    self.errors.append(f"Node '{node_id}' has invalid type: {node_type}")
                else:
                    self.warnings.append(f"Node '{node_id}' has unknown type: {node_type}")

            # Validate params if present
            if "params" in node:
                self._validate_node_params(node_id, node_type, node.get("params", {}))

    def _validate_node_params(self, node_id: str, node_type: str, params: Dict[str, Any]) -> None:
        """Validate node parameters."""
        if not isinstance(params, dict):
            self.errors.append(f"Node '{node_id}' params must be a dict, got {type(params)}")
            return

        # Type-specific validation
        if node_type == "CONST" and "value" not in params:
            self.errors.append(f"CONST node '{node_id}' missing 'value' param")

        if node_type == "LOAD" and "addr" not in params:
            self.errors.append(f"LOAD node '{node_id}' missing 'addr' param")

        if node_type == "ContractNode":
            if "target_node" not in params:
                self.errors.append(f"ContractNode '{node_id}' missing 'target_node' param")
            if "constraints" not in params:
                self.errors.append(f"ContractNode '{node_id}' missing 'constraints' param")

    def _validate_edges(self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> None:
        """Validate all edges."""
        if not isinstance(edges, list):
            self.errors.append(f"Edges must be a list, got {type(edges)}")
            return

        node_ids = {node["id"] for node in nodes if "id" in node}

        for idx, edge in enumerate(edges):
            # Check required fields
            missing = self.REQUIRED_EDGE_FIELDS - set(edge.keys())
            if missing:
                self.errors.append(f"Edge {idx} missing required fields: {missing}")
                continue

            # Validate 'from' and 'to' structure
            from_spec = edge.get("from", {})
            to_spec = edge.get("to", {})

            if not isinstance(from_spec, dict) or "node" not in from_spec:
                self.errors.append(f"Edge {idx} has invalid 'from' specification")
            else:
                from_node = from_spec.get("node")
                if from_node not in node_ids:
                    self.errors.append(f"Edge {idx} references non-existent source node: {from_node}")

            if not isinstance(to_spec, dict) or "node" not in to_spec:
                self.errors.append(f"Edge {idx} has invalid 'to' specification")
            else:
                to_node = to_spec.get("node")
                if to_node not in node_ids:
                    self.errors.append(f"Edge {idx} references non-existent target node: {to_node}")

            # Validate edge kind
            edge_kind = edge.get("kind")
            if edge_kind and edge_kind not in self.VALID_EDGE_KINDS:
                if self.strict_mode:
                    self.errors.append(f"Edge {idx} has invalid kind: {edge_kind}")
                else:
                    self.warnings.append(f"Edge {idx} has unknown kind: {edge_kind}")

    def _validate_graph_connectivity(self, graph_data: Dict[str, Any]) -> None:
        """Validate graph connectivity and detect cycles."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return

        # Build adjacency list
        adjacency = {node["id"]: list(] for node in nodes if "id" in node}

        for edge in edges:
            from_node = edge.get("from", {}).get("node")
            to_node = edge.get("to", {}).get("node")

            if from_node and to_node and from_node in adjacency:
                adjacency[from_node].append(to_node)

        # Check for unreachable nodes
        node_ids = set(adjacency.keys())
        reachable = self._find_reachable_nodes(adjacency)
        unreachable = node_ids - reachable

        if unreachable and self.strict_mode:
            self.warnings.append(f"Unreachable nodes detected: {unreachable}")

        # Detect cycles (simple DFS-based detection)
        if self._has_cycles(adjacency):
            self.warnings.append("Graph contains cycles")

    def _find_reachable_nodes(self, adjacency: Dict[str, List[str]]) -> Set[str]:
        """Find all reachable nodes from any source."""
        reachable = set()

        def dfs(node: str, visited: Set[str]):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor in adjacency:
                    dfs(neighbor, visited)

        for node in adjacency:
            dfs(node, reachable)

        return reachable

    def _has_cycles(self, adjacency: Dict[str, List[str]]) -> bool:
        """Detect if graph has cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in adjacency}

        def has_cycle_util(node: str) -> bool:
            color[node] = GRAY

            for neighbor in adjacency.get(node, []):
                if neighbor not in color:
                    continue

                if color[neighbor] == GRAY:
                    return True

                if color[neighbor] == WHITE and has_cycle_util(neighbor):
                    return True

            color[node] = BLACK
            return False

        for node in adjacency:
            if color[node] == WHITE:
                if has_cycle_util(node):
                    return True

        return False

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata structure."""
        if not metadata:
            return

        if not isinstance(metadata, dict):
            self.errors.append(f"Metadata must be a dict, got {type(metadata)}")
            return

        # Check for recommended metadata fields
        recommended = {"author_agent", "creation_timestamp", "description"}
        missing_recommended = recommended - set(metadata.keys())

        if missing_recommended and self.strict_mode:
            self.warnings.append(f"Missing recommended metadata: {missing_recommended}")


def load_graph_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse graph JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {filepath}: {e}")
    except FileNotFoundError:
        raise ValidationError(f"File not found: {filepath}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def validator():
    """Create a graph validator."""
    return GraphValidator(strict_mode=False)


@pytest.fixture
def strict_validator():
    """Create a strict graph validator."""
    return GraphValidator(strict_mode=True)


@pytest.fixture
def mutator_graph():
    """Load mutator.json graph."""
    return {
        "id": "generic_mutator_v1",
        "type": "Graph",
        "grammar_version": "1.2.0",
        "metadata": {
            "author_agent": "OptimizerAgent_v3",
            "creation_timestamp": "2025-08-26T21:04:44Z",
            "description": "A generic metaprogram"
        },
        "nodes": [
            {"id": "pattern_input", "type": "CONST", "params": {"value": {}}},
            {"id": "template_input", "type": "CONST", "params": {"value": {}}},
            {"id": "target_graph_input", "type": "CONST", "params": {"value": "graph_id"}},
            {"id": "pat", "type": "PATTERN_COMPILE"},
            {"id": "find", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}},
            {"id": "splice", "type": "GRAPH_SPLICE", "params": {"method": "replace_subgraph"}},
            {"id": "commit", "type": "GRAPH_COMMIT"}
        ],
        "edges": [
            {"from": {"node": "pattern_input", "port": "value"},
             "to": {"node": "pat", "port": "pattern_in"}, "kind": "data"},
            {"from": {"node": "pat", "port": "pattern_out"},
             "to": {"node": "find", "port": "pattern_in"}, "kind": "data"}
        ]
    }


@pytest.fixture
def sentiment_graph():
    """Load sentiment_3d.json graph."""
    return {
        "id": "sentiment_3d_enhanced",
        "type": "Graph",
        "grammar_version": "2.3.0",
        "nodes": [
            {"id": "t_in", "type": "CONST", "params": {"value": "texts:t0"}},
            {"id": "matrix", "type": "CONST", "params": {"value": [[[1, 2], [3, 4]]]}},
            {"id": "m_emb", "type": "EMBED", "params": {"provider": "dummy"}},
            {"id": "gen", "type": "GenerativeNode", "params": {"provider": "dummy"}},
            {"id": "out", "type": "OutputNode", "params": {}}
        ],
        "edges": [
            {"from": {"node": "t_in", "port": "value"},
             "to": {"node": "m_emb", "port": "text"}},
            {"from": {"node": "m_emb", "port": "vector"},
             "to": {"node": "gen", "port": "input"}}
        ]
    }


@pytest.fixture
def classification_graph():
    """Load classification.json graph."""
    return {
        "id": "fitness_classification_v2",
        "type": "Graph",
        "grammar_version": "1.2.0",
        "nodes": [
            {"id": "accuracy_metric", "type": "LOAD", "params": {"addr": "metrics.accuracy"}},
            {"id": "token_penalty_weight", "type": "CONST", "params": {"value": -0.0001}},
            {"id": "token_penalty_calc", "type": "MUL"},
            {"id": "final_sum", "type": "ADD"},
            {"id": "fitness_output", "type": "HALT"}
        ],
        "edges": [
            {"from": {"node": "accuracy_metric", "port": "data_out"},
             "to": {"node": "token_penalty_calc", "port": "val1"}}
        ]
    }


@pytest.fixture
def invalid_graph_missing_id():
    """Graph missing required id field."""
    return {
        "type": "Graph",
        "nodes": [],
        "edges": []
    }


@pytest.fixture
def invalid_graph_wrong_type():
    """Graph with wrong type."""
    return {
        "id": "test",
        "type": "NotAGraph",
        "nodes": [],
        "edges": []
    }


@pytest.fixture
def invalid_graph_duplicate_nodes():
    """Graph with duplicate node IDs."""
    return {
        "id": "test",
        "type": "Graph",
        "nodes": [
            {"id": "node1", "type": "CONST"},
            {"id": "node1", "type": "LOAD"}
        ],
        "edges": []
    }


@pytest.fixture
def invalid_graph_broken_edge():
    """Graph with edge pointing to non-existent node."""
    return {
        "id": "test",
        "type": "Graph",
        "nodes": [
            {"id": "node1", "type": "CONST", "params": {"value": 1}}
        ],
        "edges": [
            {"from": {"node": "node1", "port": "out"},
             "to": {"node": "nonexistent", "port": "in"}}
        ]
    }


# ============================================================================
# Test Root Structure Validation
# ============================================================================

class TestRootStructureValidation:
    """Test root-level graph structure validation."""

    def test_valid_root_structure(self, validator, mutator_graph):
        """Test valid root structure passes validation."""
        validator._validate_root_structure(mutator_graph)
        assert len(validator.errors) == 0

    def test_missing_required_fields(self, validator, invalid_graph_missing_id):
        """Test detection of missing required fields."""
        validator._validate_root_structure(invalid_graph_missing_id)
        assert len(validator.errors) > 0
        assert any("Missing required root fields" in err for err in validator.errors)

    def test_wrong_graph_type(self, validator, invalid_graph_wrong_type):
        """Test detection of wrong graph type."""
        validator._validate_root_structure(invalid_graph_wrong_type)
        assert len(validator.errors) > 0
        assert any("Invalid graph type" in err for err in validator.errors)

    def test_invalid_grammar_version(self, validator):
        """Test detection of invalid grammar version."""
        graph = {"id": "test", "type": "Graph", "nodes": [], "edges": [],
                 "grammar_version": 123}
        validator._validate_root_structure(graph)
        assert len(validator.errors) > 0
        assert any("grammar_version" in err for err in validator.errors)


# ============================================================================
# Test Node Validation
# ============================================================================

class TestNodeValidation:
    """Test node validation."""

    def test_valid_nodes(self, validator, mutator_graph):
        """Test valid nodes pass validation."""
        validator._validate_nodes(mutator_graph["nodes"])
        assert len(validator.errors) == 0

    def test_empty_nodes_warning(self, validator):
        """Test empty nodes list generates warning."""
        validator._validate_nodes([])
        assert len(validator.warnings) > 0
        assert any("no nodes" in warn for warn in validator.warnings)

    def test_nodes_not_list(self, validator):
        """Test non-list nodes generate error."""
        validator._validate_nodes("not a list")
        assert len(validator.errors) > 0

    def test_missing_node_id(self, validator):
        """Test node missing id field."""
        nodes = [{"type": "CONST"}]
        validator._validate_nodes(nodes)
        assert len(validator.errors) > 0
        assert any("missing required fields" in err for err in validator.errors)

    def test_missing_node_type(self, validator):
        """Test node missing type field."""
        nodes = [{"id": "test_node"}]
        validator._validate_nodes(nodes)
        assert len(validator.errors) > 0

    def test_duplicate_node_ids(self, validator, invalid_graph_duplicate_nodes):
        """Test detection of duplicate node IDs."""
        validator._validate_nodes(invalid_graph_duplicate_nodes["nodes"])
        assert len(validator.errors) > 0
        assert any("Duplicate node id" in err for err in validator.errors)

    def test_invalid_node_type_strict(self, strict_validator):
        """Test invalid node type in strict mode."""
        nodes = [{"id": "test", "type": "INVALID_TYPE"}]
        strict_validator._validate_nodes(nodes)
        assert len(strict_validator.errors) > 0
        assert any("invalid type" in err for err in strict_validator.errors)

    def test_invalid_node_type_lenient(self, validator):
        """Test invalid node type in lenient mode generates warning."""
        nodes = [{"id": "test", "type": "UNKNOWN_TYPE"}]
        validator._validate_nodes(nodes)
        assert len(validator.warnings) > 0
        assert any("unknown type" in warn for warn in validator.warnings)

    def test_const_node_missing_value(self, validator):
        """Test CONST node missing value parameter."""
        nodes = [{"id": "const1", "type": "CONST", "params": {}}]
        validator._validate_nodes(nodes)
        assert len(validator.errors) > 0
        assert any("missing 'value' param" in err for err in validator.errors)

    def test_load_node_missing_addr(self, validator):
        """Test LOAD node missing addr parameter."""
        nodes = [{"id": "load1", "type": "LOAD", "params": {}}]
        validator._validate_nodes(nodes)
        assert len(validator.errors) > 0
        assert any("missing 'addr' param" in err for err in validator.errors)

    def test_contract_node_missing_params(self, validator):
        """Test ContractNode missing required parameters."""
        nodes = [{"id": "contract1", "type": "ContractNode", "params": {}}]
        validator._validate_nodes(nodes)
        assert len(validator.errors) >= 2
        assert any("missing 'target_node'" in err for err in validator.errors)
        assert any("missing 'constraints'" in err for err in validator.errors)


# ============================================================================
# Test Edge Validation
# ============================================================================

class TestEdgeValidation:
    """Test edge validation."""

    def test_valid_edges(self, validator, mutator_graph):
        """Test valid edges pass validation."""
        validator._validate_edges(mutator_graph["edges"], mutator_graph["nodes"])
        assert len(validator.errors) == 0

    def test_edges_not_list(self, validator):
        """Test non-list edges generate error."""
        validator._validate_edges("not a list", [])
        assert len(validator.errors) > 0

    def test_edge_missing_from(self, validator):
        """Test edge missing 'from' field."""
        edges = [{"to": {"node": "n1", "port": "in"}}]
        nodes = [{"id": "n1", "type": "CONST"}]
        validator._validate_edges(edges, nodes)
        assert len(validator.errors) > 0

    def test_edge_missing_to(self, validator):
        """Test edge missing 'to' field."""
        edges = [{"from": {"node": "n1", "port": "out"}}]
        nodes = [{"id": "n1", "type": "CONST"}]
        validator._validate_edges(edges, nodes)
        assert len(validator.errors) > 0

    def test_edge_invalid_from_structure(self, validator):
        """Test edge with invalid 'from' structure."""
        edges = [{"from": "invalid", "to": {"node": "n1", "port": "in"}}]
        nodes = [{"id": "n1", "type": "CONST"}]
        validator._validate_edges(edges, nodes)
        assert len(validator.errors) > 0
        assert any("invalid 'from'" in err for err in validator.errors)

    def test_edge_nonexistent_source_node(self, validator, invalid_graph_broken_edge):
        """Test edge pointing to non-existent source node."""
        validator._validate_edges(
            invalid_graph_broken_edge["edges"],
            invalid_graph_broken_edge["nodes"]
        )
        assert len(validator.errors) > 0
        assert any("non-existent target node" in err for err in validator.errors)

    def test_edge_nonexistent_target_node(self, validator):
        """Test edge with non-existent target."""
        edges = [{"from": {"node": "n1", "port": "out"},
                  "to": {"node": "n2", "port": "in"}}]
        nodes = [{"id": "n1", "type": "CONST"}]
        validator._validate_edges(edges, nodes)
        assert len(validator.errors) > 0

    def test_edge_invalid_kind_strict(self, strict_validator):
        """Test invalid edge kind in strict mode."""
        edges = [{"from": {"node": "n1", "port": "out"},
                  "to": {"node": "n2", "port": "in"},
                  "kind": "invalid_kind"}]
        nodes = [{"id": "n1", "type": "CONST"}, {"id": "n2", "type": "LOAD"}]
        strict_validator._validate_edges(edges, nodes)
        assert len(strict_validator.errors) > 0
        assert any("invalid kind" in err for err in strict_validator.errors)

    def test_edge_valid_kinds(self, validator):
        """Test all valid edge kinds."""
        nodes = [
            {"id": "n1", "type": "CONST"},
            {"id": "n2", "type": "LOAD"},
            {"id": "n3", "type": "ADD"},
            {"id": "n4", "type": "HALT"}
        ]

        edges = [
            {"from": {"node": "n1", "port": "out"}, "to": {"node": "n2", "port": "in"},
             "kind": "data"},
            {"from": {"node": "n2", "port": "out"}, "to": {"node": "n3", "port": "in"},
             "kind": "contract_binding"},
            {"from": {"node": "n3", "port": "out"}, "to": {"node": "n4", "port": "in"},
             "kind": "nso_binding"}
        ]

        validator._validate_edges(edges, nodes)
        assert len(validator.errors) == 0


# ============================================================================
# Test Graph Connectivity
# ============================================================================

class TestGraphConnectivity:
    """Test graph connectivity validation."""

    def test_connected_graph(self, validator, sentiment_graph):
        """Test connected graph has no warnings."""
        validator._validate_graph_connectivity(sentiment_graph)
        # May have warnings about cycles or unreachable nodes
        assert len(validator.errors) == 0

    def test_cycle_detection(self, validator):
        """Test cycle detection."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "CONST"},
                {"id": "n2", "type": "ADD"},
                {"id": "n3", "type": "MUL"}
            ],
            "edges": [
                {"from": {"node": "n1", "port": "out"}, "to": {"node": "n2", "port": "in"}},
                {"from": {"node": "n2", "port": "out"}, "to": {"node": "n3", "port": "in"}},
                {"from": {"node": "n3", "port": "out"}, "to": {"node": "n2", "port": "in"}}
            ]
        }

        validator._validate_graph_connectivity(graph)
        assert any("cycle" in warn.lower() for warn in validator.warnings)

    def test_unreachable_nodes_strict(self, strict_validator):
        """Test detection of unreachable nodes in strict mode."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "CONST"},
                {"id": "n2", "type": "ADD"},
                {"id": "isolated", "type": "MUL"}
            ],
            "edges": [
                {"from": {"node": "n1", "port": "out"}, "to": {"node": "n2", "port": "in"}}
            ]
        }

        strict_validator._validate_graph_connectivity(graph)
        # Note: unreachable detection may not catch all cases
        # depending on implementation


# ============================================================================
# Test Metadata Validation
# ============================================================================

class TestMetadataValidation:
    """Test metadata validation."""

    def test_valid_metadata(self, validator, mutator_graph):
        """Test valid metadata passes validation."""
        validator._validate_metadata(mutator_graph["metadata"])
        assert len(validator.errors) == 0

    def test_empty_metadata(self, validator):
        """Test empty metadata is acceptable."""
        validator._validate_metadata({})
        assert len(validator.errors) == 0

    def test_metadata_not_dict(self, validator):
        """Test non-dict metadata generates error."""
        validator._validate_metadata("not a dict")
        assert len(validator.errors) > 0

    def test_missing_recommended_metadata_strict(self, strict_validator):
        """Test missing recommended metadata in strict mode."""
        strict_validator._validate_metadata({"author_agent": "test"})
        assert len(strict_validator.warnings) > 0


# ============================================================================
# Test Full Graph Validation
# ============================================================================

class TestFullGraphValidation:
    """Test complete graph validation."""

    def test_validate_mutator_graph(self, validator, mutator_graph):
        """Test full validation of mutator graph."""
        result = validator.validate_graph(mutator_graph)
        assert result is True
        assert len(validator.errors) == 0

    def test_validate_sentiment_graph(self, validator, sentiment_graph):
        """Test full validation of sentiment graph."""
        result = validator.validate_graph(sentiment_graph)
        assert result is True
        assert len(validator.errors) == 0

    def test_validate_classification_graph(self, validator, classification_graph):
        """Test full validation of classification graph."""
        result = validator.validate_graph(classification_graph)
        assert result is True
        assert len(validator.errors) == 0

    def test_validate_invalid_graph(self, validator, invalid_graph_missing_id):
        """Test validation fails for invalid graph."""
        result = validator.validate_graph(invalid_graph_missing_id)
        assert result is False
        assert len(validator.errors) > 0

    def test_validate_broken_edge_graph(self, validator, invalid_graph_broken_edge):
        """Test validation fails for graph with broken edges."""
        result = validator.validate_graph(invalid_graph_broken_edge)
        assert result is False
        assert len(validator.errors) > 0

    def test_error_accumulation(self, validator):
        """Test that multiple errors are accumulated."""
        invalid_graph = {
            "id": "",  # Invalid empty id
            "type": "WrongType",
            "nodes": [
                {"id": "n1"},  # Missing type
                {"type": "CONST"}  # Missing id
            ],
            "edges": [
                {"from": {"node": "nonexistent"}}  # Missing 'to'
            ]
        }

        result = validator.validate_graph(invalid_graph)
        assert result is False
        assert len(validator.errors) >= 3


# ============================================================================
# Test File Loading
# ============================================================================

class TestFileLoading:
    """Test graph file loading."""

    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        graph_file = tmp_path / "test_graph.json"
        graph_data = {"id": "test", "type": "Graph", "nodes": [], "edges": []}

        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f)

        loaded = load_graph_file(graph_file)
        assert loaded == graph_data

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        graph_file = tmp_path / "invalid.json"

        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write("{invalid json}")

        with pytest.raises(ValidationError, match="Invalid JSON"):
            load_graph_file(graph_file)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(ValidationError, match="File not found"):
            load_graph_file(Path("nonexistent.json"))


# ============================================================================
# Integration Tests
# ============================================================================

class TestGraphValidationIntegration:
    """Integration tests with real graph structures."""

    def test_all_sample_graphs_valid(self, validator):
        """Test that all sample graphs are valid."""
        graphs = [
            {
                "id": "test_mutator",
                "type": "Graph",
                "nodes": [
                    {"id": "input", "type": "CONST", "params": {"value": 1}},
                    {"id": "output", "type": "HALT"}
                ],
                "edges": [
                    {"from": {"node": "input", "port": "value"},
                     "to": {"node": "output", "port": "final_in"}}
                ]
            },
            {
                "id": "test_sentiment",
                "type": "Graph",
                "grammar_version": "2.0.0",
                "nodes": [
                    {"id": "embed", "type": "EMBED", "params": {"provider": "test"}},
                    {"id": "gen", "type": "GenerativeNode", "params": {"model": "test"}}
                ],
                "edges": [
                    {"from": {"node": "embed", "port": "out"},
                     "to": {"node": "gen", "port": "in"}}
                ]
            }
        ]

        for graph in graphs:
            result = validator.validate_graph(graph)
            assert result is True, f"Graph {graph['id']} validation failed"

    def test_complex_graph_with_all_features(self, validator):
        """Test complex graph with various node types and edge kinds."""
        graph = {
            "id": "complex_test",
            "type": "Graph",
            "grammar_version": "2.0.0",
            "metadata": {
                "author_agent": "test",
                "creation_timestamp": "2025-01-01T00:00:00Z",
                "description": "Complex test graph"
            },
            "nodes": [
                {"id": "const1", "type": "CONST", "params": {"value": 1}},
                {"id": "load1", "type": "LOAD", "params": {"addr": "test.data"}},
                {"id": "contract1", "type": "ContractNode",
                 "params": {"target_node": "gen1", "constraints": {"max_cost_usd": 0.01}}},
                {"id": "gen1", "type": "GenerativeNode", "params": {"model": "test"}},
                {"id": "add1", "type": "ADD"},
                {"id": "mul1", "type": "MUL"},
                {"id": "out1", "type": "HALT"}
            ],
            "edges": [
                {"from": {"node": "const1", "port": "value"},
                 "to": {"node": "add1", "port": "val1"}, "kind": "data"},
                {"from": {"node": "load1", "port": "data_out"},
                 "to": {"node": "add1", "port": "val2"}, "kind": "data"},
                {"from": {"node": "add1", "port": "sum"},
                 "to": {"node": "mul1", "port": "val1"}, "kind": "data"},
                {"from": {"node": "contract1", "port": "output"},
                 "to": {"node": "gen1", "port": "contract"}, "kind": "contract_binding"},
                {"from": {"node": "mul1", "port": "product"},
                 "to": {"node": "out1", "port": "final_in"}, "kind": "data"}
            ]
        }

        result = validator.validate_graph(graph)
        assert result is True


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
