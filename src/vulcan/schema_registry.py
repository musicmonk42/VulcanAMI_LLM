"""
Schema Registry Module - Industry Standard Schema Validation for Vulcan

This module provides a centralized, thread-safe registry for JSON Schema validation
across the entire Vulcan system. It follows industry-standard patterns including:
- Thread-safe singleton pattern
- Lazy loading of schemas
- Comprehensive error handling
- Detailed validation error reporting
- Type hints and documentation
- Graceful degradation

Author: Vulcan Engineering Team
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Conditional import of jsonschema for validation
try:
    import jsonschema
    from jsonschema import Draft7Validator, validators
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    Draft7Validator = None
    validators = None

logger = logging.getLogger(__name__)

# Import schema auto-generator for converting EBNF to JSON Schema
_SCHEMA_AUTO_GENERATOR_AVAILABLE = False
try:
    # Try relative import first (when installed as package)
    from ...tools import schema_auto_generator as sag
    _SCHEMA_AUTO_GENERATOR_AVAILABLE = True
except (ImportError, ValueError):
    # Fall back to path manipulation for development/testing
    try:
        tools_path = Path(__file__).resolve().parent.parent.parent / "tools"
        if tools_path.exists() and str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
        import schema_auto_generator as sag
        _SCHEMA_AUTO_GENERATOR_AVAILABLE = True
    except ImportError as e:
        logger.warning(
            f"schema_auto_generator not available: {e}. "
            "Schema generation from EBNF will be disabled."
        )
        sag = None


# Default EBNF grammar definitions for Vulcan schemas
VULCAN_SCHEMAS = {
    "observation": """
        <Observation> ::= timestamp:NUMBER, domain:STRING, variables:json_object, 
                          [confidence:FLOAT], [source:STRING], [metadata:json_object];
    """,
    "prediction": """
        <Prediction> ::= target:STRING, value:json_value, confidence:FLOAT, 
                         method:STRING, [timestamp:NUMBER];
    """,
    "agent_task": """
        <AgentTask> ::= task_id:ID, task_type:STRING, query:STRING,
                        [context:json_object], [tools:STRING*], priority:INT;
    """,
    "dqs_components": """
        <DQSComponents> ::= pii_confidence:FLOAT, graph_completeness:FLOAT,
                            syntactic_completeness:FLOAT;
    """,
    "cost_config": """
        <CostConfig> ::= tool_name:STRING, time_ms:NUMBER, energy_mj:NUMBER,
                         [cold_start_penalty:NUMBER], [health_threshold:FLOAT];
    """,
    "reasoning_query": """
        <ReasoningQuery> ::= query:STRING, query_type:STRING, complexity:FLOAT,
                             [context:json_object], [constraints:json_object];
    """,
}


@dataclass
class ValidationError:
    """
    Represents a schema validation error with detailed context.
    
    Attributes:
        message: Human-readable error message
        path: JSON path where the error occurred
        schema_path: Path in the schema that failed
        validator: The validator that failed (e.g., 'type', 'required')
        instance: The actual value that failed validation
    """
    message: str
    path: List[str] = field(default_factory=list)
    schema_path: List[str] = field(default_factory=list)
    validator: str = ""
    instance: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "path": ".".join(str(p) for p in self.path) if self.path else "root",
            "schema_path": ".".join(str(p) for p in self.schema_path) if self.schema_path else "root",
            "validator": self.validator,
            "instance": str(self.instance) if self.instance is not None else None,
        }


@dataclass
class ValidationResult:
    """
    Result of schema validation.
    
    Attributes:
        valid: Whether the data passed validation
        errors: List of validation errors if validation failed
        schema_name: Name of the schema that was used
        timestamp: When validation was performed
    """
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    schema_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "errors": [err.to_dict() for err in self.errors],
            "schema_name": self.schema_name,
            "timestamp": self.timestamp.isoformat(),
            "error_count": len(self.errors),
        }


class SchemaRegistry:
    """
    Thread-safe singleton registry for JSON Schema validation.
    
    This class provides centralized schema management with:
    - Lazy loading of default schemas from EBNF definitions
    - Thread-safe registration and retrieval of schemas
    - Comprehensive validation with detailed error reporting
    - Graceful degradation when dependencies are unavailable
    
    Example:
        >>> registry = SchemaRegistry.get_instance()
        >>> result = registry.validate(data, "observation")
        >>> if not result.valid:
        ...     logger.error(f"Validation failed: {result.errors}")
    """
    
    _instance: Optional[SchemaRegistry] = None
    _lock: threading.Lock = threading.Lock()
    
    def __init__(self):
        """Initialize the schema registry. Use get_instance() instead."""
        if SchemaRegistry._instance is not None:
            raise RuntimeError(
                "SchemaRegistry is a singleton. Use SchemaRegistry.get_instance()"
            )
        
        # Thread-safe storage for compiled schemas
        self._schemas: Dict[str, Any] = {}
        self._schemas_lock = threading.Lock()
        
        # Track which schemas have been loaded
        self._loaded_schemas: Set[str] = set()
        
        # Track validation statistics
        self._validation_count = 0
        self._validation_failures = 0
        
        # Check if jsonschema is available
        if not JSONSCHEMA_AVAILABLE:
            logger.warning(
                "jsonschema library not available. Schema validation will be disabled. "
                "Install with: pip install jsonschema"
            )
        
        logger.info("SchemaRegistry initialized")
    
    @classmethod
    def get_instance(cls) -> SchemaRegistry:
        """
        Get the singleton instance of SchemaRegistry (thread-safe).
        
        Returns:
            The singleton SchemaRegistry instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance. Used primarily for testing.
        
        Warning: This should only be used in test scenarios.
        """
        with cls._lock:
            cls._instance = None
            logger.debug("SchemaRegistry instance reset")
    
    def _compile_ebnf_to_schema(self, ebnf_grammar: str) -> Optional[Dict[str, Any]]:
        """
        Compile EBNF grammar to JSON Schema using schema_auto_generator.
        
        Args:
            ebnf_grammar: EBNF grammar definition
            
        Returns:
            Compiled JSON Schema dict or None if compilation fails
        """
        if not _SCHEMA_AUTO_GENERATOR_AVAILABLE:
            logger.error("Cannot compile EBNF: schema_auto_generator not available")
            return None
        
        try:
            # Parse EBNF and build schema
            productions = sag.parse_ebnf(ebnf_grammar.strip())
            schema = sag.build_json_schema_from_productions(productions, strict=False)
            
            logger.debug(f"Successfully compiled EBNF to JSON Schema")
            return schema
        except Exception as e:
            logger.error(f"Failed to compile EBNF grammar: {e}", exc_info=True)
            return None
    
    def _load_default_schema(self, schema_name: str) -> bool:
        """
        Lazy load a default schema from EBNF definition.
        
        Args:
            schema_name: Name of the schema to load
            
        Returns:
            True if schema was loaded successfully, False otherwise
        """
        if schema_name in self._loaded_schemas:
            return True
        
        if schema_name not in VULCAN_SCHEMAS:
            logger.warning(f"Schema '{schema_name}' not found in default schemas")
            return False
        
        ebnf_grammar = VULCAN_SCHEMAS[schema_name]
        schema = self._compile_ebnf_to_schema(ebnf_grammar)
        
        if schema is None:
            logger.error(f"Failed to compile default schema: {schema_name}")
            return False
        
        with self._schemas_lock:
            self._schemas[schema_name] = schema
            self._loaded_schemas.add(schema_name)
        
        logger.info(f"Loaded default schema: {schema_name}")
        return True
    
    def register_schema(
        self, 
        name: str, 
        grammar: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new schema from either EBNF grammar or JSON Schema.
        
        Args:
            name: Name to register the schema under
            grammar: EBNF grammar definition (mutually exclusive with schema)
            schema: JSON Schema dict (mutually exclusive with grammar)
            
        Returns:
            True if schema was registered successfully, False otherwise
            
        Raises:
            ValueError: If both grammar and schema are provided, or neither
        """
        if (grammar is None and schema is None) or (grammar is not None and schema is not None):
            raise ValueError("Must provide exactly one of 'grammar' or 'schema'")
        
        compiled_schema = None
        
        if grammar is not None:
            compiled_schema = self._compile_ebnf_to_schema(grammar)
            if compiled_schema is None:
                logger.error(f"Failed to register schema '{name}': compilation failed")
                return False
        else:
            compiled_schema = schema
        
        # Validate the schema itself if jsonschema is available
        if JSONSCHEMA_AVAILABLE:
            try:
                Draft7Validator.check_schema(compiled_schema)
            except Exception as e:
                logger.error(f"Invalid schema structure for '{name}': {e}")
                return False
        
        with self._schemas_lock:
            self._schemas[name] = compiled_schema
            self._loaded_schemas.add(name)
        
        logger.info(f"Registered schema: {name}")
        return True
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a compiled schema by name.
        
        Lazily loads default schemas if not already loaded.
        
        Args:
            name: Name of the schema to retrieve
            
        Returns:
            The compiled schema dict, or None if not found
        """
        with self._schemas_lock:
            if name in self._schemas:
                return self._schemas[name]
        
        # Try to lazy load from defaults
        if name in VULCAN_SCHEMAS:
            if self._load_default_schema(name):
                with self._schemas_lock:
                    return self._schemas.get(name)
        
        logger.warning(f"Schema not found: {name}")
        return None
    
    def list_schemas(self) -> List[str]:
        """
        List all available schema names (both loaded and default).
        
        Returns:
            List of schema names
        """
        with self._schemas_lock:
            loaded = set(self._schemas.keys())
        
        default_names = set(VULCAN_SCHEMAS.keys())
        all_schemas = loaded | default_names
        
        return sorted(list(all_schemas))
    
    def validate(
        self, 
        data: Any, 
        schema_name: str,
        raise_on_error: bool = False
    ) -> ValidationResult:
        """
        Validate data against a named schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            raise_on_error: If True, raise exception on validation failure
            
        Returns:
            ValidationResult with validation status and any errors
            
        Raises:
            ValueError: If schema not found or validation fails (when raise_on_error=True)
        """
        self._validation_count += 1
        
        # Check if jsonschema is available
        if not JSONSCHEMA_AVAILABLE:
            logger.warning(
                f"Schema validation skipped for '{schema_name}': jsonschema not available"
            )
            return ValidationResult(
                valid=True,  # Gracefully degrade
                schema_name=schema_name,
                errors=[],
            )
        
        # Get the schema
        schema = self.get_schema(schema_name)
        if schema is None:
            error_msg = f"Schema not found: {schema_name}"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return ValidationResult(
                valid=False,
                schema_name=schema_name,
                errors=[ValidationError(message=error_msg)],
            )
        
        # Perform validation
        try:
            validator = Draft7Validator(schema)
            errors_list = []
            
            for error in validator.iter_errors(data):
                val_error = ValidationError(
                    message=error.message,
                    path=list(error.absolute_path),
                    schema_path=list(error.absolute_schema_path),
                    validator=error.validator,
                    instance=error.instance,
                )
                errors_list.append(val_error)
            
            if errors_list:
                self._validation_failures += 1
                result = ValidationResult(
                    valid=False,
                    errors=errors_list,
                    schema_name=schema_name,
                )
                
                # Log validation failures
                logger.warning(
                    f"Validation failed for schema '{schema_name}': "
                    f"{len(errors_list)} error(s)"
                )
                for err in errors_list:
                    logger.debug(f"  - {err.message} at {'.'.join(str(p) for p in err.path)}")
                
                if raise_on_error:
                    raise ValueError(
                        f"Validation failed: {len(errors_list)} error(s). "
                        f"First error: {errors_list[0].message}"
                    )
                
                return result
            else:
                logger.debug(f"Validation succeeded for schema '{schema_name}'")
                return ValidationResult(
                    valid=True,
                    schema_name=schema_name,
                    errors=[],
                )
        
        except Exception as e:
            self._validation_failures += 1
            error_msg = f"Validation error for schema '{schema_name}': {e}"
            logger.error(error_msg, exc_info=True)
            
            if raise_on_error:
                raise ValueError(error_msg) from e
            
            return ValidationResult(
                valid=False,
                schema_name=schema_name,
                errors=[ValidationError(message=error_msg)],
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            "total_validations": self._validation_count,
            "failed_validations": self._validation_failures,
            "success_rate": (
                1.0 - (self._validation_failures / self._validation_count)
                if self._validation_count > 0
                else 1.0
            ),
            "loaded_schemas": len(self._loaded_schemas),
            "available_schemas": len(self.list_schemas()),
        }


# Convenience function for quick validation
def validate_data(
    data: Any,
    schema_name: str,
    raise_on_error: bool = False
) -> ValidationResult:
    """
    Convenience function to validate data against a named schema.
    
    Args:
        data: Data to validate
        schema_name: Name of schema to validate against
        raise_on_error: If True, raise exception on validation failure
        
    Returns:
        ValidationResult with validation status and any errors
    """
    registry = SchemaRegistry.get_instance()
    return registry.validate(data, schema_name, raise_on_error)


__all__ = [
    "SchemaRegistry",
    "ValidationResult",
    "ValidationError",
    "validate_data",
    "VULCAN_SCHEMAS",
]
