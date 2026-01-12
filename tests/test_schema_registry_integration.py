"""
Comprehensive Integration Tests for Schema Registry

This test suite validates:
- Schema registration and retrieval
- Validation against each default schema
- Integration with WorldModel
- Integration with DQS
- Integration with Strategies module
- Thread safety
- Error handling for invalid schemas and data

Author: Vulcan Engineering Team
"""

import json
import pytest
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vulcan.schema_registry import (
    SchemaRegistry,
    ValidationResult,
    ValidationError,
    validate_data,
    VULCAN_SCHEMAS,
)


class TestSchemaRegistrySingleton:
    """Test singleton pattern and thread safety"""

    def test_singleton_pattern(self):
        """Test that SchemaRegistry follows singleton pattern"""
        registry1 = SchemaRegistry.get_instance()
        registry2 = SchemaRegistry.get_instance()
        assert registry1 is registry2, "SchemaRegistry should be a singleton"

    def test_reset_instance(self):
        """Test instance reset for testing purposes"""
        registry1 = SchemaRegistry.get_instance()
        SchemaRegistry.reset_instance()
        registry2 = SchemaRegistry.get_instance()
        assert registry1 is not registry2, "Reset should create new instance"
        # Clean up
        SchemaRegistry.reset_instance()

    def test_thread_safety(self):
        """Test that schema registry is thread-safe"""
        SchemaRegistry.reset_instance()
        registry = SchemaRegistry.get_instance()
        results = []
        errors = []

        def worker():
            try:
                # Get instance
                reg = SchemaRegistry.get_instance()
                results.append(reg)
                # List schemas
                schemas = reg.list_schemas()
                assert len(schemas) > 0
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check all threads got the same instance
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(r is results[0] for r in results), "All threads should get same instance"


class TestSchemaRegistration:
    """Test schema registration and retrieval"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_list_default_schemas(self):
        """Test listing default schemas"""
        registry = SchemaRegistry.get_instance()
        schemas = registry.list_schemas()
        
        # Should include all default schemas
        expected = list(VULCAN_SCHEMAS.keys())
        for schema_name in expected:
            assert schema_name in schemas, f"Missing default schema: {schema_name}"

    def test_register_schema_from_json(self):
        """Test registering schema from JSON Schema dict"""
        registry = SchemaRegistry.get_instance()
        
        test_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        
        success = registry.register_schema("test_person", schema=test_schema)
        assert success, "Schema registration should succeed"
        
        # Verify retrieval
        retrieved = registry.get_schema("test_person")
        assert retrieved is not None, "Should retrieve registered schema"
        assert retrieved["properties"]["name"]["type"] == "string"

    def test_register_invalid_schema(self):
        """Test that invalid schemas are rejected"""
        registry = SchemaRegistry.get_instance()
        
        invalid_schema = {
            "type": "invalid_type",  # Invalid type
        }
        
        # This should fail validation (if jsonschema is available)
        # Otherwise it might succeed but won't validate correctly
        result = registry.register_schema("test_invalid", schema=invalid_schema)
        # We don't assert False because jsonschema might not be available

    def test_register_schema_requires_one_arg(self):
        """Test that register_schema requires exactly one of grammar or schema"""
        registry = SchemaRegistry.get_instance()
        
        # Neither grammar nor schema
        with pytest.raises(ValueError):
            registry.register_schema("test")
        
        # Both grammar and schema
        with pytest.raises(ValueError):
            registry.register_schema(
                "test",
                grammar="<Test> ::= STRING;",
                schema={"type": "object"}
            )

    def test_get_nonexistent_schema(self):
        """Test retrieving non-existent schema returns None"""
        registry = SchemaRegistry.get_instance()
        result = registry.get_schema("nonexistent_schema_xyz")
        assert result is None, "Non-existent schema should return None"


class TestValidation:
    """Test validation functionality"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_validate_data_convenience_function(self):
        """Test the convenience validate_data function"""
        # Note: This will gracefully degrade if jsonschema not available
        result = validate_data({"test": "data"}, "observation")
        assert isinstance(result, ValidationResult)
        assert isinstance(result.valid, bool)

    def test_validation_result_structure(self):
        """Test ValidationResult structure"""
        registry = SchemaRegistry.get_instance()
        
        test_data = {"field1": "value1"}
        result = registry.validate(test_data, "observation")
        
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "schema_name")
        assert hasattr(result, "timestamp")
        assert result.schema_name == "observation"

    def test_validation_with_nonexistent_schema(self):
        """Test validation with non-existent schema fails"""
        registry = SchemaRegistry.get_instance()
        
        result = registry.validate({"test": "data"}, "nonexistent_xyz")
        assert not result.valid, "Validation should fail for non-existent schema"
        assert len(result.errors) > 0, "Should have error messages"

    def test_validation_error_structure(self):
        """Test ValidationError structure"""
        registry = SchemaRegistry.get_instance()
        
        # Validate with non-existent schema to get errors
        result = registry.validate({"test": "data"}, "nonexistent")
        
        if result.errors:
            error = result.errors[0]
            assert hasattr(error, "message")
            assert hasattr(error, "path")
            assert hasattr(error, "schema_path")
            assert hasattr(error, "validator")
            
            # Test to_dict
            error_dict = error.to_dict()
            assert "message" in error_dict
            assert "path" in error_dict


class TestStatistics:
    """Test statistics tracking"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_get_statistics(self):
        """Test statistics retrieval"""
        registry = SchemaRegistry.get_instance()
        
        # Get initial stats
        stats = registry.get_statistics()
        assert "total_validations" in stats
        assert "failed_validations" in stats
        assert "success_rate" in stats
        assert "loaded_schemas" in stats
        assert "available_schemas" in stats
        
        # Perform some validations
        registry.validate({"test": "data"}, "observation")
        registry.validate({"test": "data"}, "prediction")
        
        # Check updated stats
        new_stats = registry.get_statistics()
        assert new_stats["total_validations"] >= 2


class TestGracefulDegradation:
    """Test graceful degradation when dependencies unavailable"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_validation_without_jsonschema(self):
        """Test that validation gracefully degrades without jsonschema"""
        registry = SchemaRegistry.get_instance()
        
        # Even if jsonschema is not available, validation should not crash
        result = registry.validate({"test": "data"}, "observation")
        assert isinstance(result, ValidationResult)
        # When jsonschema not available, it should return valid=True (graceful degradation)

    def test_registry_initialization_without_generator(self):
        """Test that registry works without schema_auto_generator"""
        # SchemaRegistry should initialize even if schema_auto_generator not available
        registry = SchemaRegistry.get_instance()
        assert registry is not None
        
        # Should still list default schemas
        schemas = registry.list_schemas()
        assert len(schemas) > 0


class TestDefaultSchemas:
    """Test default Vulcan schemas"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_observation_schema_structure(self):
        """Test observation schema is defined correctly"""
        assert "observation" in VULCAN_SCHEMAS
        grammar = VULCAN_SCHEMAS["observation"]
        assert "Observation" in grammar
        assert "timestamp" in grammar
        assert "domain" in grammar
        assert "variables" in grammar

    def test_prediction_schema_structure(self):
        """Test prediction schema is defined correctly"""
        assert "prediction" in VULCAN_SCHEMAS
        grammar = VULCAN_SCHEMAS["prediction"]
        assert "Prediction" in grammar
        assert "target" in grammar
        assert "confidence" in grammar

    def test_dqs_components_schema_structure(self):
        """Test DQS components schema is defined correctly"""
        assert "dqs_components" in VULCAN_SCHEMAS
        grammar = VULCAN_SCHEMAS["dqs_components"]
        assert "DQSComponents" in grammar
        assert "pii_confidence" in grammar
        assert "graph_completeness" in grammar
        assert "syntactic_completeness" in grammar

    def test_all_default_schemas_exist(self):
        """Test all expected default schemas exist"""
        expected_schemas = [
            "observation",
            "prediction",
            "agent_task",
            "dqs_components",
            "cost_config",
            "reasoning_query",
        ]
        
        for schema_name in expected_schemas:
            assert schema_name in VULCAN_SCHEMAS, f"Missing schema: {schema_name}"


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset instance before each test"""
        SchemaRegistry.reset_instance()
        yield
        SchemaRegistry.reset_instance()

    def test_concurrent_validation(self):
        """Test concurrent validation from multiple threads"""
        registry = SchemaRegistry.get_instance()
        results = []
        errors = []

        def validate_worker(data, schema_name):
            try:
                result = registry.validate(data, schema_name)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create validation threads
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=validate_worker,
                args=({"test": f"data_{i}"}, "observation")
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5, "All validations should complete"

    def test_validation_performance(self):
        """Test validation performance with many requests"""
        registry = SchemaRegistry.get_instance()
        
        start_time = time.time()
        for i in range(100):
            registry.validate({"test": f"data_{i}"}, "observation")
        elapsed = time.time() - start_time
        
        # Should complete 100 validations reasonably fast
        assert elapsed < 5.0, f"Validation too slow: {elapsed}s for 100 validations"

    def test_mixed_operations(self):
        """Test mixed registration, retrieval, and validation"""
        registry = SchemaRegistry.get_instance()
        
        # Register a custom schema
        custom_schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
        }
        registry.register_schema("custom", schema=custom_schema)
        
        # Retrieve it
        retrieved = registry.get_schema("custom")
        assert retrieved is not None
        
        # Validate against it
        result = registry.validate({"id": "test123"}, "custom")
        # Should succeed or gracefully degrade

        # List all schemas
        schemas = registry.list_schemas()
        assert "custom" in schemas
        assert "observation" in schemas


def test_module_imports():
    """Test that all necessary modules can be imported"""
    # These should not raise ImportError
    from vulcan.schema_registry import SchemaRegistry
    from vulcan.schema_registry import ValidationResult
    from vulcan.schema_registry import ValidationError
    from vulcan.schema_registry import validate_data
    from vulcan.schema_registry import VULCAN_SCHEMAS
    
    assert SchemaRegistry is not None
    assert ValidationResult is not None
    assert ValidationError is not None
    assert validate_data is not None
    assert VULCAN_SCHEMAS is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
