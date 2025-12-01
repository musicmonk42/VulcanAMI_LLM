"""
Comprehensive pytest suite for graph_validator.py
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock # Added patch

# Import the module to test - use full path since short aliasing is disabled
from unified_runtime import graph_validator as gv


class TestResourceLimits:
    """Test ResourceLimits constants"""

    def test_resource_limits_defined(self):
        """Test that all resource limits are defined"""
        assert gv.ResourceLimits.MAX_EXECUTION_TIME_S > 0
        assert gv.ResourceLimits.MAX_NODE_COUNT > 0
        assert gv.ResourceLimits.MAX_EDGE_COUNT > 0
        assert gv.ResourceLimits.MAX_GRAPH_DEPTH > 0
        assert gv.ResourceLimits.MAX_RECURSION_DEPTH > 0

    def test_memory_limits(self):
        """Test memory limit values"""
        assert gv.ResourceLimits.MAX_MEMORY_MB == 8000
        assert gv.ResourceLimits.MAX_MEMORY_PER_NODE_MB == 1000

    def test_data_limits(self):
        """Test data size limits"""
        assert gv.ResourceLimits.MAX_STRING_LENGTH > 0
        assert gv.ResourceLimits.MAX_ARRAY_LENGTH > 0


class TestValidationError:
    """Test ValidationError enum"""

    def test_error_types(self):
        """Test all error types are defined"""
        assert gv.ValidationError.STRUCTURE_INVALID.value == "STRUCTURE_INVALID"
        assert gv.ValidationError.RESOURCE_EXCEEDED.value == "RESOURCE_EXCEEDED"
        assert gv.ValidationError.CYCLE_DETECTED.value == "CYCLE_DETECTED"
        assert gv.ValidationError.NODE_INVALID.value == "NODE_INVALID"
        assert gv.ValidationError.EDGE_INVALID.value == "EDGE_INVALID"
        assert gv.ValidationError.SECURITY_VIOLATION.value == "SECURITY_VIOLATION"


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_result_creation(self):
        """Test creating validation result"""
        result = gv.ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.metadata) == 0

    def test_result_with_errors(self):
        """Test result with errors"""
        result = gv.ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"]
        )

        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_result_to_dict(self):
        """Test converting result to dict"""
        result = gv.ValidationResult(
            is_valid=True,
            warnings=["Warning 1"]
        )

        d = result.to_dict()
        assert d['valid'] is True
        assert 'errors' in d
        assert 'warnings' in d
        assert len(d['warnings']) == 1


class TestGraphValidator:
    """Test GraphValidator class"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        # Ensure tests not needing special ontology load the real one if available
        return gv.GraphValidator()

    @pytest.fixture
    def simple_graph(self):
        """Simple valid graph"""
        return {
            "nodes": [
                {"id": "n1", "type": "Source"},
                {"id": "n2", "type": "Process"}
            ],
            "edges": [
                {"from": "n1", "to": "n2"}
            ]
        }

    def test_validator_creation(self, validator):
        """Test creating validator"""
        assert validator.max_node_count == gv.ResourceLimits.MAX_NODE_COUNT
        assert validator.max_edge_count == gv.ResourceLimits.MAX_EDGE_COUNT
        assert validator.enable_cycle_detection is True

    def test_validate_simple_graph(self, validator, simple_graph):
        """Test validating simple graph"""
        result = validator.validate_graph(simple_graph)

        assert isinstance(result, gv.ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_not_dict(self, validator):
        """Test validation fails for non-dict input"""
        result = validator.validate_graph("not a dict")

        assert result.is_valid is False
        assert any("dictionary" in err.lower() for err in result.errors)

    def test_validate_empty_graph(self, validator):
        """Test validation of empty graph"""
        result = validator.validate_graph({"nodes": [], "edges": []})

        # Should be valid but with warning
        assert result.is_valid is True
        assert any("no nodes" in warn.lower() for warn in result.warnings)

    def test_validate_too_many_nodes(self, validator):
        """Test validation fails with too many nodes"""
        large_graph = {
            "nodes": [{"id": f"n{i}", "type": "Test"} for i in range(validator.max_node_count + 1)],
            "edges": []
        }

        result = validator.validate_graph(large_graph)

        assert result.is_valid is False
        assert any("too many nodes" in err.lower() for err in result.errors)

    def test_validate_too_many_edges(self, validator):
        """Test validation fails with too many edges"""
        graph = {
            "nodes": [{"id": "n1", "type": "Test"}, {"id": "n2", "type": "Test"}],
            "edges": [{"from": "n1", "to": "n2"} for _ in range(validator.max_edge_count + 1)]
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("too many edges" in err.lower() for err in result.errors)

    def test_validate_nodes_not_list(self, validator):
        """Test validation fails when nodes is not a list"""
        result = validator.validate_graph({"nodes": "not a list", "edges": []})

        assert result.is_valid is False
        assert any("must be a list" in err.lower() for err in result.errors)

    def test_validate_node_missing_id(self, validator):
        """Test validation fails for node without ID"""
        graph = {
            "nodes": [{"type": "Test"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("missing 'id'" in err.lower() for err in result.errors)

    def test_validate_node_missing_type(self, validator):
        """Test validation fails for node without type"""
        graph = {
            "nodes": [{"id": "n1"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("missing 'type'" in err.lower() for err in result.errors)

    def test_validate_duplicate_node_id(self, validator):
        """Test validation fails for duplicate node IDs"""
        graph = {
            "nodes": [
                {"id": "n1", "type": "Test"},
                {"id": "n1", "type": "Test"}
            ],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("duplicate" in err.lower() for err in result.errors)

    def test_validate_dangerous_node_type(self, validator):
        """Test warning for dangerous node types"""
        graph = {
            "nodes": [{"id": "n1", "type": "ExecuteNode"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        # Should be valid but with warning
        assert any("dangerous" in warn.lower() for warn in result.warnings)

    def test_validate_edge_missing_from(self, validator):
        """Test validation fails for edge without from"""
        graph = {
            "nodes": [{"id": "n1", "type": "Test"}],
            "edges": [{"to": "n1"}]
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("missing 'from'" in err.lower() for err in result.errors)

    def test_validate_edge_unknown_node(self, validator):
        """Test validation fails for edge referencing unknown node"""
        graph = {
            "nodes": [{"id": "n1", "type": "Test"}],
            "edges": [{"from": "n1", "to": "unknown"}]
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("unknown" in err.lower() for err in result.errors)

    def test_validate_params_too_long_string(self, validator):
        """Test validation fails for overly long string param"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {"key": "x" * (gv.ResourceLimits.MAX_STRING_LENGTH + 1)}
            }],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False # Should be an error
        assert any("too long" in err.lower() for err in result.errors)

    def test_validate_params_injection_pattern(self, validator):
        """Test validation catches code injection patterns"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {"code": "eval('malicious code')"}
            }],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("dangerous pattern" in err.lower() for err in result.errors)

    def test_detect_cycles_simple_cycle(self, validator):
        """Test cycle detection with simple cycle"""
        graph = {
            "nodes": [
                {"id": "n1", "type": "Test"},
                {"id": "n2", "type": "Test"}
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n1"}
            ]
        }

        result = validator.validate_graph(graph)

        # Should warn about cycles
        assert result.metadata['has_cycles'] is True
        assert any("cycle" in warn.lower() for warn in result.warnings)

    def test_detect_no_cycles(self, validator):
        """Test cycle detection with DAG"""
        graph = {
            "nodes": [
                {"id": "n1", "type": "Test"},
                {"id": "n2", "type": "Test"},
                {"id": "n3", "type": "Test"}
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"}
            ]
        }

        # Use validate_graph to get metadata
        result = validator.validate_graph(graph)
        assert result.metadata['has_cycles'] is False
        assert not any("cycle" in warn.lower() for warn in result.warnings)

    def test_extract_node_reference_string(self, validator):
        """Test extracting node reference from string"""
        ref = validator._extract_node_reference("n1")
        assert ref == "n1"

    def test_extract_node_reference_dict(self, validator):
        """Test extracting node reference from dict"""
        ref = validator._extract_node_reference({"node": "n1", "port": "out"})
        assert ref == "n1"

    def test_extract_node_reference_none(self, validator):
        """Test extracting node reference from invalid input"""
        ref = validator._extract_node_reference(None)
        assert ref is None
        ref_empty_str = validator._extract_node_reference("")
        assert ref_empty_str is None
        ref_empty_dict = validator._extract_node_reference({})
        assert ref_empty_dict is None
        ref_invalid_dict = validator._extract_node_reference({"node": ""})
        assert ref_invalid_dict is None

    def test_estimate_memory_usage(self, validator):
        """Test memory usage estimation"""
        graph = {
            "nodes": [{"id": f"n{i}", "type": "Test"} for i in range(10)],
            "edges": [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(9)]
        }

        memory_mb = validator._estimate_memory_usage(graph)
        assert memory_mb > 0

    def test_check_resources(self, validator, simple_graph):
        """Test resource checking"""
        result = gv.ValidationResult(is_valid=True)
        validator._check_resources(simple_graph, result)

        assert 'estimated_memory_mb' in result.metadata

    def test_security_validation_file_access(self, validator):
        """Test security validation detects file access"""
        graph = {
            "nodes": [{"id": "n1", "type": "FileNode"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert any("file system" in warn.lower() for warn in result.warnings)

    def test_security_validation_network_access(self, validator):
        """Test security validation detects network access"""
        graph = {
            "nodes": [{"id": "n1", "type": "NetworkNode"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert any("network" in warn.lower() for warn in result.warnings)

    def test_security_validation_code_execution(self, validator):
        """Test security validation blocks code execution"""
        graph = {
            "nodes": [{"id": "n1", "type": "ExecNode"}],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is False
        assert any("execution" in err.lower() for err in result.errors)

    def test_suspicious_content_path_traversal(self, validator):
        """Test detection of path traversal"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {"path": "../../../etc/passwd"}
            }],
            "edges": []
        }

        result = validator.validate_graph(graph)

        assert any("path traversal" in warn.lower() for warn in result.warnings)

    def test_suspicious_content_absolute_path(self, validator):
        """Test detection of absolute paths"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {"path": "/absolute/path"} # Unix style
            },
            {
                "id": "n2",
                "type": "Test",
                "params": {"path": "C:\\Windows\\System32"} # Windows style
            }
            ],
            "edges": []
        }

        result = validator.validate_graph(graph)

        # Check that at least one warning about absolute path exists
        assert any("absolute path" in warn.lower() for warn in result.warnings)

    def test_metadata_includes_counts(self, validator, simple_graph):
        """Test metadata includes node and edge counts"""
        result = validator.validate_graph(simple_graph)

        assert result.metadata['node_count'] == 2
        assert result.metadata['edge_count'] == 1

    def test_validator_with_disabled_features(self):
        """Test validator with features disabled"""
        validator = gv.GraphValidator(
            enable_cycle_detection=False,
            enable_resource_checking=False,
            enable_security_validation=False
        )

        # Graph with cycle and dangerous node type
        graph = {
            "nodes": [{"id": "n1", "type": "ExecuteNode"}, {"id": "n2", "type": "Test"}],
            "edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n1"}]
        }

        result = validator.validate_graph(graph)

        # Should be valid and have no cycle/security warnings when disabled
        assert result.is_valid is True
        assert not any("cycle" in w.lower() for w in result.warnings)
        assert not any("dangerous" in w.lower() for w in result.warnings)
        assert 'estimated_memory_mb' not in result.metadata # Resource check disabled
        assert result.metadata['has_cycles'] is False # Cycle detection disabled


class TestSemanticValidation:
    """Test semantic validation with ontology"""

    @pytest.fixture
    def validator_with_ontology(self, tmp_path):
        """Create validator with test ontology, mocking _load_ontology"""
        test_ontology_data = {
            "concepts": {
                "SourceNode": {
                    "allowed_properties": ["value", "output_type"]
                },
                "ProcessNode": {
                    "allowed_properties": ["operation", "params"]
                },
                "OutputNode": { # Added for completeness based on relationships
                    "allowed_properties": []
                }
            },
            "relationships": [
                {"source": "SourceNode", "target": "ProcessNode"},
                {"source": "ProcessNode", "target": "OutputNode"}
            ]
        }

        # Patch the _load_ontology method for the duration of this fixture
        # Target the specific module where GraphValidator is defined
        with patch('graph_validator.GraphValidator._load_ontology', return_value=test_ontology_data) as mock_load:
            try:
                # Now when GraphValidator is initialized, its _load_ontology will return test_ontology_data
                validator = gv.GraphValidator()
                # Ensure the mock loaded the correct data (optional sanity check)
                assert validator.ontology == test_ontology_data
                yield validator
            finally:
                 pass # Context manager handles unpatching automatically

    def test_semantic_validation_valid_types(self, validator_with_ontology):
        """Test semantic validation with valid node types"""
        # No need for skip check, fixture ensures ontology is mocked
        # if validator_with_ontology.ontology is None:
        #     pytest.skip("Ontology not loaded")

        graph = {
            "nodes": [
                {"id": "n1", "type": "SourceNode"},
                {"id": "n2", "type": "ProcessNode"}
            ],
            "edges": [{"from": "n1", "to": "n2"}]
        }

        result = validator_with_ontology.validate_graph(graph)

        # Should be valid with no semantic warnings related to types/connections
        assert result.is_valid is True
        assert not any("not defined" in w.lower() for w in result.warnings)
        assert not any("unusual pattern" in w.lower() for w in result.warnings)
        assert not any("unexpected parameter" in w.lower() for w in result.warnings)

    def test_semantic_validation_unknown_type(self, validator_with_ontology):
        """Test semantic validation warns about unknown types"""
        # if validator_with_ontology.ontology is None:
        #     pytest.skip("Ontology not loaded")

        graph = {
            "nodes": [{"id": "n1", "type": "UnknownNode"}],
            "edges": []
        }

        result = validator_with_ontology.validate_graph(graph)

        # Should warn about unknown type
        assert result.is_valid is True # Warnings don't invalidate
        assert any("not defined" in warn.lower() for warn in result.warnings)

    def test_semantic_validation_invalid_property(self, validator_with_ontology):
        """Test semantic validation warns about invalid properties"""
        # if validator_with_ontology.ontology is None:
        #     pytest.skip("Ontology not loaded")

        graph = {
            "nodes": [{
                "id": "n1",
                "type": "SourceNode",
                "params": {"invalid_prop": "value"} # 'invalid_prop' is not in allowed_properties
            }],
            "edges": []
        }

        result = validator_with_ontology.validate_graph(graph)

        # Should warn about unexpected parameter
        assert result.is_valid is True # Warnings don't invalidate
        assert any("unexpected parameter" in warn.lower() for warn in result.warnings)

    def test_semantic_validation_invalid_connection(self, validator_with_ontology):
        """Test semantic validation warns about invalid connections"""
        # if validator_with_ontology.ontology is None:
        #     pytest.skip("Ontology not loaded")

        graph = {
            "nodes": [
                {"id": "n1", "type": "ProcessNode"},
                {"id": "n2", "type": "SourceNode"} # Invalid: Process -> Source not in relationships
            ],
            "edges": [{"from": "n1", "to": "n2"}]
        }

        result = validator_with_ontology.validate_graph(graph)

        # Should warn about unusual connection
        assert result.is_valid is True # Warnings don't invalidate
        assert any("unusual pattern" in warn.lower() for warn in result.warnings)


class TestComplexGraphs:
    """Test validation of complex graph structures"""

    @pytest.fixture
    def validator(self):
        # Use default validator (loads real ontology if available)
        return gv.GraphValidator()

    def test_validate_diamond_graph(self, validator):
        """Test validation of diamond-shaped graph"""
        graph = {
            "nodes": [
                {"id": "source", "type": "Source"},
                {"id": "left", "type": "Process"},
                {"id": "right", "type": "Process"},
                {"id": "sink", "type": "Sink"}
            ],
            "edges": [
                {"from": "source", "to": "left"},
                {"from": "source", "to": "right"},
                {"from": "left", "to": "sink"},
                {"from": "right", "to": "sink"}
            ]
        }

        result = validator.validate_graph(graph)

        assert result.is_valid is True
        assert result.metadata.get('has_cycles') is False # Check metadata explicitly

    def test_validate_nested_params(self, validator):
        """Test validation of deeply nested parameters"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {
                    "level1": {
                        "level2": {
                            "level3": {
                                "value": "deep"
                            }
                        }
                    }
                }
            }],
            "edges": []
        }

        result = validator.validate_graph(graph)

        # Should handle nested params without error
        assert result.is_valid is True

    def test_validate_large_array_param(self, validator):
        """Test validation fails with large array parameter"""
        graph = {
            "nodes": [{
                "id": "n1",
                "type": "Test",
                "params": {
                    # Create a list slightly larger than the limit
                    "data": list(range(gv.ResourceLimits.MAX_ARRAY_LENGTH + 1))
                }
            }],
            "edges": []
        }

        result = validator.validate_graph(graph)

        # Should fail due to array length
        assert result.is_valid is False
        assert any("too long" in err.lower() for err in result.errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])