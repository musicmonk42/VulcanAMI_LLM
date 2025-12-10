#!/usr/bin/env python3
"""
Comprehensive test suite for Graphix graph validation against core ontology
Tests validation of graphs against graphix_core_ontology.json

Run with:
    pytest test_ontology_validation.py -v
    pytest test_ontology_validation.py -v --cov=ontology_validator --cov-report=html
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

# ============================================================================
# Ontology-Based Validator Implementation
# ============================================================================

class OntologyValidationError(Exception):
    """Ontology validation error."""
    pass


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]

    def add_error(self, msg: str):
        """Add error message."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        """Add warning message."""
        self.warnings.append(msg)

    def add_info(self, msg: str):
        """Add info message."""
        self.info.append(msg)


class OntologyValidator:
    """Validates graphs against Graphix core ontology."""

    def __init__(self, ontology_path: Optional[Path] = None):
        """
        Initialize validator with ontology.

        Args:
            ontology_path: Path to ontology JSON file
        """
        self.ontology = None
        self.node_types = set()
        self.edge_types = set()
        self.semantic_types = set()
        self.lifecycle_statuses = set()

        if ontology_path:
            self.load_ontology(ontology_path)

    def load_ontology(self, ontology_path: Path):
        """Load ontology from file."""
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                self.ontology = json.load(f)

            # Extract valid types
            self.node_types = set(self.ontology.get("node_types", {}).keys())
            self.edge_types = set(self.ontology.get("edge_types", {}).keys())
            self.semantic_types = set(self.ontology.get("semantic_types", {}).keys())
            self.lifecycle_statuses = set(self.ontology.get("lifecycle_statuses", {}).keys())

        except FileNotFoundError:
            raise OntologyValidationError(f"Ontology file not found: {ontology_path}")
        except json.JSONDecodeError as e:
            raise OntologyValidationError(f"Invalid JSON in ontology: {e}")

    def validate_graph(self, graph: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete graph against ontology.

        Args:
            graph: Graph dictionary to validate

        Returns:
            ValidationResult with errors, warnings, and info
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        if not self.ontology:
            result.add_error("Ontology not loaded")
            return result

        # Validate root structure
        self._validate_root_structure(graph, result)

        # Validate nodes
        nodes = graph.get("nodes", [])
        self._validate_nodes(nodes, result)

        # Validate edges
        edges = graph.get("edges", [])
        self._validate_edges(edges, nodes, result)

        # Validate grammar version
        self._validate_grammar_version(graph, result)

        # Validate metadata
        self._validate_metadata(graph.get("metadata", {}), result)

        return result

    def _validate_root_structure(self, graph: Dict[str, Any], result: ValidationResult):
        """Validate root graph structure."""
        required_fields = {"id", "type", "nodes", "edges"}
        missing = required_fields - set(graph.keys())

        if missing:
            result.add_error(f"Missing required root fields: {missing}")

        if graph.get("type") != "Graph":
            result.add_error(f"Graph type must be 'Graph', got '{graph.get('type')}'")

        if not isinstance(graph.get("nodes"), list):
            result.add_error("'nodes' must be a list")

        if not isinstance(graph.get("edges"), list):
            result.add_error("'edges' must be a list")

    def _validate_nodes(self, nodes: List[Dict[str, Any]], result: ValidationResult):
        """Validate all nodes against ontology."""
        if not nodes:
            result.add_warning("Graph has no nodes")
            return

        node_ids = set()

        for idx, node in enumerate(nodes):
            # Check required fields
            if "id" not in node:
                result.add_error(f"Node {idx} missing 'id' field")
                continue

            node_id = node["id"]

            if "type" not in node:
                result.add_error(f"Node '{node_id}' missing 'type' field")
                continue

            node_type = node["type"]

            # Check for duplicates
            if node_id in node_ids:
                result.add_error(f"Duplicate node ID: '{node_id}'")
            else:
                node_ids.add(node_id)

            # Validate against ontology
            self._validate_node_type(node_id, node_type, node, result)

    def _validate_node_type(self, node_id: str, node_type: str,
                           node: Dict[str, Any], result: ValidationResult):
        """Validate node type against ontology."""
        if node_type not in self.node_types:
            result.add_error(
                f"Node '{node_id}' has unknown type '{node_type}'. "
                f"Valid types: {sorted(list(self.node_types)[:10])}..."
            )
            return

        # Get ontology definition
        type_def = self.ontology["node_types"][node_type]

        # Check lifecycle status
        lifecycle = type_def.get("lifecycle_status")
        if lifecycle == "deprecated":
            result.add_warning(
                f"Node '{node_id}' uses deprecated type '{node_type}'"
            )
        elif lifecycle == "experimental":
            result.add_info(
                f"Node '{node_id}' uses experimental type '{node_type}'"
            )
        elif lifecycle == "superseded":
            result.add_error(
                f"Node '{node_id}' uses superseded type '{node_type}' - "
                "this type should not be used"
            )

        # Validate node-specific requirements
        self._validate_node_params(node_id, node_type, node.get("params", {}), result)

    def _validate_node_params(self, node_id: str, node_type: str,
                              params: Dict[str, Any], result: ValidationResult):
        """Validate node parameters based on type."""
        # Type-specific validation rules
        rules = {
            "CONST": {"required": ["value"]},
            "LOAD": {"required": ["addr"]},
            "ContractNode": {"required": ["target_node", "constraints"]},
            "EMBED": {"required": ["provider"]},
            "GenerativeNode": {"required": ["provider"]},
            "PATTERN_COMPILE": {"required": []},
            "FIND_SUBGRAPH": {"required": []},
            "GRAPH_SPLICE": {"required": []},
            "NSO_MODIFY": {"required": []},
            "ETHICAL_LABEL": {"required": []},
            "EVAL": {"required": []},
            "HALT": {"required": []},
        }

        if node_type in rules:
            required = rules[node_type]["required"]
            missing = set(required) - set(params.keys())

            if missing:
                result.add_error(
                    f"Node '{node_id}' of type '{node_type}' missing "
                    f"required params: {missing}"
                )

        # Validate ContractNode constraints
        if node_type == "ContractNode":
            constraints = params.get("constraints", {})
            if not isinstance(constraints, dict):
                result.add_error(
                    f"Node '{node_id}' ContractNode constraints must be a dict"
                )
            else:
                # Check for reasonable constraint keys
                valid_constraint_keys = {
                    "max_cost_usd", "max_latency_ms", "accuracy_floor",
                    "max_compute_cycles", "min_accuracy", "max_tokens"
                }
                for key in constraints.keys():
                    if key not in valid_constraint_keys:
                        result.add_warning(
                            f"Node '{node_id}' has unknown constraint key: '{key}'"
                        )

    def _validate_edges(self, edges: List[Dict[str, Any]],
                       nodes: List[Dict[str, Any]], result: ValidationResult):
        """Validate all edges against ontology."""
        if not edges:
            result.add_warning("Graph has no edges")
            return

        node_ids = {node["id"] for node in nodes if "id" in node}

        for idx, edge in enumerate(edges):
            # Check required fields
            if "from" not in edge or "to" not in edge:
                result.add_error(f"Edge {idx} missing 'from' or 'to' field")
                continue

            from_spec = edge["from"]
            to_spec = edge["to"]

            # Validate structure
            if not isinstance(from_spec, dict) or "node" not in from_spec:
                result.add_error(f"Edge {idx} has invalid 'from' specification")
                continue

            if not isinstance(to_spec, dict) or "node" not in to_spec:
                result.add_error(f"Edge {idx} has invalid 'to' specification")
                continue

            from_node = from_spec["node"]
            to_node = to_spec["node"]

            # Check node references
            if from_node not in node_ids:
                result.add_error(
                    f"Edge {idx} references non-existent source node: '{from_node}'"
                )

            if to_node not in node_ids:
                result.add_error(
                    f"Edge {idx} references non-existent target node: '{to_node}'"
                )

            # Validate edge kind/type
            edge_kind = edge.get("kind")
            if edge_kind:
                self._validate_edge_kind(idx, edge_kind, result)

    def _validate_edge_kind(self, edge_idx: int, edge_kind: str,
                           result: ValidationResult):
        """Validate edge kind against ontology."""
        if edge_kind not in self.edge_types:
            result.add_error(
                f"Edge {edge_idx} has unknown kind '{edge_kind}'. "
                f"Valid kinds: {sorted(list(self.edge_types)[:10])}..."
            )
            return

        # Get ontology definition
        type_def = self.ontology["edge_types"][edge_kind]

        # Check lifecycle status
        lifecycle = type_def.get("lifecycle_status")
        if lifecycle == "deprecated":
            result.add_warning(
                f"Edge {edge_idx} uses deprecated kind '{edge_kind}'"
            )
        elif lifecycle == "experimental":
            result.add_info(
                f"Edge {edge_idx} uses experimental kind '{edge_kind}'"
            )

    def _validate_grammar_version(self, graph: Dict[str, Any],
                                  result: ValidationResult):
        """Validate grammar version."""
        version = graph.get("grammar_version")

        if not version:
            result.add_info("No grammar_version specified")
            return

        # Version format: X.Y.Z or X.Y.Z-suffix
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
        if not re.match(pattern, version):
            result.add_error(f"Invalid grammar_version format: '{version}'")

        # Check if version is reasonable
        ontology_version = self.ontology.get("version", "0.0.0")
        if self._compare_versions(version, ontology_version) > 0:
            result.add_warning(
                f"Graph version '{version}' is newer than ontology version "
                f"'{ontology_version}'"
            )

    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two version strings.

        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        def parse(v):
            parts = v.split('-')[0].split('.')
            return tuple(int(p) for p in parts)

        try:
            p1 = parse(v1)
            p2 = parse(v2)

            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1
            return 0
        except (ValueError, IndexError):
            return 0

    def _validate_metadata(self, metadata: Dict[str, Any], result: ValidationResult):
        """Validate metadata structure."""
        if not metadata:
            return

        if not isinstance(metadata, dict):
            result.add_error("Metadata must be a dictionary")
            return

        # Check for recommended fields
        recommended = {"author_agent", "creation_timestamp", "description"}
        has_recommended = set(metadata.keys()) & recommended

        if not has_recommended:
            result.add_info(
                f"Metadata missing recommended fields: {recommended}"
            )

        # Validate timestamp format if present
        timestamp = metadata.get("creation_timestamp")
        if timestamp:
            # ISO 8601 format check
            iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?$'
            if not re.match(iso_pattern, timestamp):
                result.add_warning(
                    f"creation_timestamp '{timestamp}' not in ISO 8601 format"
                )

    def get_node_types_by_lifecycle(self, status: str) -> List[str]:
        """Get all node types with given lifecycle status."""
        if not self.ontology:
            return []

        return [
            node_type for node_type, definition in
            self.ontology.get("node_types", {}).items()
            if definition.get("lifecycle_status") == status
        ]

    def get_edge_types_by_lifecycle(self, status: str) -> List[str]:
        """Get all edge types with given lifecycle status."""
        if not self.ontology:
            return []

        return [
            edge_type for edge_type, definition in
            self.ontology.get("edge_types", {}).items()
            if definition.get("lifecycle_status") == status
        ]

    def validate_node_type_exists(self, node_type: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if node type exists in ontology.

        Returns:
            (exists, definition)
        """
        if not self.ontology:
            return False, None

        definition = self.ontology.get("node_types", {}).get(node_type)
        return definition is not None, definition

    def validate_edge_type_exists(self, edge_type: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if edge type exists in ontology.

        Returns:
            (exists, definition)
        """
        if not self.ontology:
            return False, None

        definition = self.ontology.get("edge_types", {}).get(edge_type)
        return definition is not None, definition


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ontology_data():
    """Create sample ontology data."""
    return {
        "version": "2.4.0",
        "metadata": {
            "description": "Test ontology",
            "base_uri": "https://graphix.ai/ontology/"
        },
        "lifecycle_statuses": {
            "active": "Fully supported",
            "deprecated": "Being phased out",
            "experimental": "Under development",
            "superseded": "Replaced by another type"
        },
        "node_types": {
            "CONST": {
                "uri": "https://graphix.ai/ontology/CONST",
                "description": "Constant value node",
                "external_ref": "https://schema.org/DataFeedItem",
                "lifecycle_status": "active"
            },
            "LOAD": {
                "uri": "https://graphix.ai/ontology/LOAD",
                "description": "Load data node",
                "external_ref": "https://schema.org/DataDownload",
                "lifecycle_status": "active"
            },
            "GenerativeNode": {
                "uri": "https://graphix.ai/ontology/GenerativeNode",
                "description": "Generative AI node",
                "lifecycle_status": "active"
            },
            "DeprecatedNode": {
                "uri": "https://graphix.ai/ontology/DeprecatedNode",
                "description": "Deprecated node type",
                "lifecycle_status": "deprecated"
            },
            "SupersededNode": {
                "uri": "https://graphix.ai/ontology/SupersededNode",
                "description": "Superseded node type",
                "lifecycle_status": "superseded"
            },
            "ContractNode": {
                "uri": "https://graphix.ai/ontology/ContractNode",
                "description": "Contract definition node",
                "lifecycle_status": "active"
            },
            "OutputNode": {
                "uri": "https://graphix.ai/ontology/OutputNode",
                "description": "Output node",
                "lifecycle_status": "active"
            },
            "EMBED": {
                "uri": "https://graphix.ai/ontology/EMBED",
                "description": "Embedding node",
                "lifecycle_status": "active"
            }
        },
        "edge_types": {
            "data": {
                "uri": "https://graphix.ai/ontology/data",
                "description": "Data flow edge",
                "lifecycle_status": "active"
            },
            "control": {
                "uri": "https://graphix.ai/ontology/control",
                "description": "Control flow edge",
                "lifecycle_status": "active"
            },
            "contract_binding": {
                "uri": "https://graphix.ai/ontology/contract_binding",
                "description": "Contract binding edge",
                "lifecycle_status": "active"
            },
            "deprecated_edge": {
                "uri": "https://graphix.ai/ontology/deprecated_edge",
                "description": "Deprecated edge type",
                "lifecycle_status": "deprecated"
            }
        },
        "semantic_types": {
            "computation": {
                "uri": "https://graphix.ai/ontology/computation",
                "description": "Computational operation",
                "lifecycle_status": "active"
            }
        }
    }


@pytest.fixture
def validator(ontology_data, tmp_path):
    """Create validator with sample ontology."""
    ontology_file = tmp_path / "test_ontology.json"
    with open(ontology_file, 'w', encoding='utf-8') as f:
        json.dump(ontology_data, f, indent=2)

    validator = OntologyValidator(ontology_file)
    return validator


@pytest.fixture
def valid_graph():
    """Create valid test graph."""
    return {
        "id": "test_graph",
        "type": "Graph",
        "grammar_version": "2.3.0",
        "metadata": {
            "author_agent": "test_agent",
            "creation_timestamp": "2025-01-15T10:00:00Z",
            "description": "Test graph"
        },
        "nodes": [
            {
                "id": "const1",
                "type": "CONST",
                "params": {"value": 42}
            },
            {
                "id": "load1",
                "type": "LOAD",
                "params": {"addr": "data.json"}
            },
            {
                "id": "out1",
                "type": "OutputNode",
                "params": {}
            }
        ],
        "edges": [
            {
                "from": {"node": "const1", "port": "value"},
                "to": {"node": "load1", "port": "input"},
                "kind": "data"
            },
            {
                "from": {"node": "load1", "port": "output"},
                "to": {"node": "out1", "port": "input"}
            }
        ]
    }


# ============================================================================
# Test Validator Initialization
# ============================================================================

class TestValidatorInitialization:
    """Test validator initialization and ontology loading."""

    def test_init_without_ontology(self):
        """Test creating validator without ontology."""
        validator = OntologyValidator()
        assert validator.ontology is None
        assert len(validator.node_types) == 0
        assert len(validator.edge_types) == 0

    def test_init_with_ontology(self, validator):
        """Test creating validator with ontology."""
        assert validator.ontology is not None
        assert len(validator.node_types) > 0
        assert len(validator.edge_types) > 0
        assert "CONST" in validator.node_types
        assert "data" in validator.edge_types

    def test_load_nonexistent_ontology(self):
        """Test loading non-existent ontology file."""
        validator = OntologyValidator()
        with pytest.raises(OntologyValidationError, match="not found"):
            validator.load_ontology(Path("nonexistent.json"))

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        ontology_file = tmp_path / "invalid.json"
        with open(ontology_file, 'w') as f:
            f.write("{invalid json")

        validator = OntologyValidator()
        with pytest.raises(OntologyValidationError, match="Invalid JSON"):
            validator.load_ontology(ontology_file)


# ============================================================================
# Test Root Structure Validation
# ============================================================================

class TestRootStructureValidation:
    """Test root structure validation."""

    def test_valid_root_structure(self, validator, valid_graph):
        """Test valid root structure."""
        result = validator.validate_graph(valid_graph)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_fields(self, validator):
        """Test missing required fields."""
        graph = {"type": "Graph", "nodes": []}
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("Missing required root fields" in err for err in result.errors)

    def test_wrong_type(self, validator):
        """Test wrong graph type."""
        graph = {
            "id": "test",
            "type": "NotAGraph",
            "nodes": [],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("type must be 'Graph'" in err for err in result.errors)

    def test_nodes_not_list(self, validator):
        """Test nodes not being a list."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": "not_a_list",
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("'nodes' must be a list" in err for err in result.errors)


# ============================================================================
# Test Node Validation
# ============================================================================

class TestNodeValidation:
    """Test node validation against ontology."""

    def test_valid_nodes(self, validator, valid_graph):
        """Test valid nodes."""
        result = validator.validate_graph(valid_graph)
        assert result.is_valid

    def test_unknown_node_type(self, validator):
        """Test unknown node type."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "UnknownNodeType"}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("unknown type 'UnknownNodeType'" in err for err in result.errors)

    def test_deprecated_node_type(self, validator):
        """Test deprecated node type generates warning."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "DeprecatedNode"}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert len(result.warnings) > 0
        assert any("deprecated" in warn for warn in result.warnings)

    def test_superseded_node_type(self, validator):
        """Test superseded node type generates error."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "SupersededNode"}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("superseded" in err for err in result.errors)

    def test_duplicate_node_ids(self, validator):
        """Test duplicate node IDs."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}},
                {"id": "n1", "type": "LOAD", "params": {"addr": "test"}}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("Duplicate node ID" in err for err in result.errors)

    def test_const_node_missing_value(self, validator):
        """Test CONST node missing value parameter."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "const1", "type": "CONST", "params": {}}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("missing required params" in err and "value" in err
                  for err in result.errors)

    def test_load_node_missing_addr(self, validator):
        """Test LOAD node missing addr parameter."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "load1", "type": "LOAD", "params": {}}
            ],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("missing required params" in err and "addr" in err
                  for err in result.errors)

    def test_contract_node_validation(self, validator):
        """Test ContractNode parameter validation."""
        # Missing target_node
        graph1 = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "c1", "type": "ContractNode",
                 "params": {"constraints": {}}}
            ],
            "edges": []
        }
        result1 = validator.validate_graph(graph1)
        assert not result1.is_valid
        assert any("target_node" in err for err in result1.errors)

        # Missing constraints
        graph2 = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "c1", "type": "ContractNode",
                 "params": {"target_node": "n1"}}
            ],
            "edges": []
        }
        result2 = validator.validate_graph(graph2)
        assert not result2.is_valid
        assert any("constraints" in err for err in result2.errors)


# ============================================================================
# Test Edge Validation
# ============================================================================

class TestEdgeValidation:
    """Test edge validation against ontology."""

    def test_valid_edges(self, validator, valid_graph):
        """Test valid edges."""
        result = validator.validate_graph(valid_graph)
        assert result.is_valid

    def test_unknown_edge_kind(self, validator):
        """Test unknown edge kind."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}},
                {"id": "n2", "type": "OutputNode", "params": {}}
            ],
            "edges": [
                {
                    "from": {"node": "n1", "port": "out"},
                    "to": {"node": "n2", "port": "in"},
                    "kind": "unknown_kind"
                }
            ]
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("unknown kind" in err for err in result.errors)

    def test_deprecated_edge_kind(self, validator):
        """Test deprecated edge kind."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}},
                {"id": "n2", "type": "OutputNode", "params": {}}
            ],
            "edges": [
                {
                    "from": {"node": "n1", "port": "out"},
                    "to": {"node": "n2", "port": "in"},
                    "kind": "deprecated_edge"
                }
            ]
        }
        result = validator.validate_graph(graph)

        assert len(result.warnings) > 0
        assert any("deprecated" in warn for warn in result.warnings)

    def test_edge_missing_fields(self, validator):
        """Test edge missing required fields."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}}
            ],
            "edges": [
                {"from": {"node": "n1", "port": "out"}}
            ]
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("missing 'from' or 'to'" in err for err in result.errors)

    def test_edge_invalid_structure(self, validator):
        """Test edge with invalid structure."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}}
            ],
            "edges": [
                {
                    "from": "invalid",
                    "to": {"node": "n1", "port": "in"}
                }
            ]
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("invalid 'from' specification" in err for err in result.errors)

    def test_edge_nonexistent_node(self, validator):
        """Test edge referencing non-existent node."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": 1}}
            ],
            "edges": [
                {
                    "from": {"node": "n1", "port": "out"},
                    "to": {"node": "nonexistent", "port": "in"}
                }
            ]
        }
        result = validator.validate_graph(graph)

        assert not result.is_valid
        assert any("non-existent target node" in err for err in result.errors)


# ============================================================================
# Test Version Validation
# ============================================================================

class TestVersionValidation:
    """Test grammar version validation."""

    def test_valid_version_format(self, validator):
        """Test valid version formats."""
        valid_versions = [
            "1.0.0",
            "2.3.1",
            "10.20.30",
            "1.0.0-alpha",
            "2.3.0-beta.1",
            "3.0.0-rc.2"
        ]

        for version in valid_versions:
            graph = {
                "id": "test",
                "type": "Graph",
                "grammar_version": version,
                "nodes": [],
                "edges": []
            }
            result = validator.validate_graph(graph)
            # Should not have version format errors
            assert not any("Invalid grammar_version format" in err
                          for err in result.errors)

    def test_invalid_version_format(self, validator):
        """Test invalid version formats."""
        invalid_versions = [
            "1.0",
            "1",
            "v1.0.0",
            "1.0.0.0",
            "abc"
        ]

        for version in invalid_versions:
            graph = {
                "id": "test",
                "type": "Graph",
                "grammar_version": version,
                "nodes": [],
                "edges": []
            }
            result = validator.validate_graph(graph)
            assert any("Invalid grammar_version format" in err
                      for err in result.errors)

    def test_version_newer_than_ontology(self, validator):
        """Test version newer than ontology version."""
        graph = {
            "id": "test",
            "type": "Graph",
            "grammar_version": "999.0.0",
            "nodes": [],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert any("newer than ontology version" in warn
                  for warn in result.warnings)


# ============================================================================
# Test Metadata Validation
# ============================================================================

class TestMetadataValidation:
    """Test metadata validation."""

    def test_valid_metadata(self, validator, valid_graph):
        """Test valid metadata."""
        result = validator.validate_graph(valid_graph)
        assert result.is_valid

    def test_missing_metadata(self, validator):
        """Test graph without metadata."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [],
            "edges": []
        }
        result = validator.validate_graph(graph)
        # Should be valid but may have info messages
        assert result.is_valid

    def test_invalid_timestamp_format(self, validator):
        """Test invalid timestamp format."""
        graph = {
            "id": "test",
            "type": "Graph",
            "metadata": {
                "creation_timestamp": "not a timestamp"
            },
            "nodes": [],
            "edges": []
        }
        result = validator.validate_graph(graph)

        assert any("not in ISO 8601 format" in warn for warn in result.warnings)


# ============================================================================
# Test Lifecycle Status Queries
# ============================================================================

class TestLifecycleQueries:
    """Test lifecycle status query methods."""

    def test_get_deprecated_node_types(self, validator):
        """Test getting deprecated node types."""
        deprecated = validator.get_node_types_by_lifecycle("deprecated")
        assert "DeprecatedNode" in deprecated

    def test_get_active_node_types(self, validator):
        """Test getting active node types."""
        active = validator.get_node_types_by_lifecycle("active")
        assert "CONST" in active
        assert "LOAD" in active
        assert "DeprecatedNode" not in active

    def test_get_experimental_node_types(self, validator):
        """Test getting experimental node types."""
        experimental = validator.get_node_types_by_lifecycle("experimental")
        # Should be empty in test ontology
        assert len(experimental) == 0

    def test_get_deprecated_edge_types(self, validator):
        """Test getting deprecated edge types."""
        deprecated = validator.get_edge_types_by_lifecycle("deprecated")
        assert "deprecated_edge" in deprecated


# ============================================================================
# Test Type Existence Checks
# ============================================================================

class TestTypeExistenceChecks:
    """Test type existence checking methods."""

    def test_node_type_exists(self, validator):
        """Test checking if node type exists."""
        exists, definition = validator.validate_node_type_exists("CONST")
        assert exists
        assert definition is not None
        assert definition["lifecycle_status"] == "active"

    def test_node_type_not_exists(self, validator):
        """Test checking non-existent node type."""
        exists, definition = validator.validate_node_type_exists("NonExistent")
        assert not exists
        assert definition is None

    def test_edge_type_exists(self, validator):
        """Test checking if edge type exists."""
        exists, definition = validator.validate_edge_type_exists("data")
        assert exists
        assert definition is not None

    def test_edge_type_not_exists(self, validator):
        """Test checking non-existent edge type."""
        exists, definition = validator.validate_edge_type_exists("nonexistent")
        assert not exists
        assert definition is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWithRealGraphs:
    """Integration tests with real graph structures."""

    def test_complex_valid_graph(self, validator):
        """Test complex valid graph."""
        graph = {
            "id": "complex_test",
            "type": "Graph",
            "grammar_version": "2.3.0",
            "metadata": {
                "author_agent": "test",
                "creation_timestamp": "2025-01-15T10:00:00Z",
                "description": "Complex test"
            },
            "nodes": [
                {"id": "input", "type": "CONST", "params": {"value": "test"}},
                {"id": "embed", "type": "EMBED", "params": {"provider": "test"}},
                {"id": "gen", "type": "GenerativeNode", "params": {"provider": "test"}},
                {"id": "contract", "type": "ContractNode",
                 "params": {
                     "target_node": "gen",
                     "constraints": {"max_cost_usd": 0.01}
                 }},
                {"id": "output", "type": "OutputNode", "params": {}}
            ],
            "edges": [
                {
                    "from": {"node": "input", "port": "value"},
                    "to": {"node": "embed", "port": "input"},
                    "kind": "data"
                },
                {
                    "from": {"node": "embed", "port": "output"},
                    "to": {"node": "gen", "port": "input"},
                    "kind": "data"
                },
                {
                    "from": {"node": "contract", "port": "output"},
                    "to": {"node": "gen", "port": "contract"},
                    "kind": "contract_binding"
                },
                {
                    "from": {"node": "gen", "port": "output"},
                    "to": {"node": "output", "port": "input"},
                    "kind": "data"
                }
            ]
        }

        result = validator.validate_graph(graph)
        assert result.is_valid
        assert len(result.errors) == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
