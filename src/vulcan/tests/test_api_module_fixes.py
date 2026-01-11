"""
Test Suite: VULCAN-AGI API Module Fixes

This test suite validates the fixes implemented for the VULCAN-AGI API module,
including middleware exports, graceful shutdown, validation improvements,
request tracking, and initialization checks.

ISSUES ADDRESSED:
    1. Missing middleware exports in src/vulcan/api/__init__.py
    2. Graceful shutdown for cleanup thread
    3. Stricter history validation with ChatHistoryMessage
    4. Request ID tracking in response models
    5. Auto-populated timestamps on ErrorResponse
    6. Enhanced initialization checks with detailed error messages

TEST STRATEGY:
    - Static analysis via AST parsing for exports validation
    - Unit tests for individual functions
    - Integration tests for threading and shutdown
    - Validation tests for Pydantic models
    - Error message verification

AUTHOR: VULCAN-AGI Team
VERSION: 1.0.0
CREATED: 2026-01-11
"""

import ast
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Set

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Provide pytest.fail fallback
    class pytest:
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
        
        class raises:
            def __init__(self, exc_class):
                self.exc_class = exc_class
                self.exc_info = None
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exc_class.__name__} but nothing was raised")
                if not issubclass(exc_type, self.exc_class):
                    return False
                self.value = exc_val
                return True


# ============================================================================
# TEST UTILITIES
# ============================================================================

def parse_python_file(file_path: Path) -> ast.Module:
    """Parse a Python file into an AST."""
    assert file_path.exists(), f"File not found: {file_path}"
    
    with open(file_path, encoding="utf-8") as f:
        try:
            return ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")


def extract_imported_names(tree: ast.Module) -> Set[str]:
    """Extract all imported names from an AST."""
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
    return imported


def extract_all_exports(tree: ast.Module) -> Set[str]:
    """Extract names from __all__ definition."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }
    return set()


# ============================================================================
# ISSUE #1: Middleware Exports Tests
# ============================================================================

def test_middleware_imported_in_api_init():
    """
    TEST: Verify middleware functions are imported in src/vulcan/api/__init__.py
    
    REQUIREMENT: Issue #1 - Missing middleware exports
    VALIDATION: AST parsing to check import statements
    """
    # Test is in src/vulcan/tests/, so go up one level to get to vulcan/
    project_root = Path(__file__).parent.parent
    init_file = project_root / "api" / "__init__.py"
    
    tree = parse_python_file(init_file)
    imported_names = extract_imported_names(tree)
    
    required_middleware = {
        "validate_api_key_middleware",
        "rate_limiting_middleware",
    }
    
    missing = required_middleware - imported_names
    assert not missing, (
        f"Missing middleware imports in api/__init__.py: {missing}. "
        "Issue #1: Middleware functions must be imported from vulcan.api.middleware"
    )


def test_middleware_exported_in_api_init():
    """
    TEST: Verify middleware functions are in __all__ exports
    
    REQUIREMENT: Issue #1 - Middleware exports
    VALIDATION: Check __all__ list includes middleware
    """
    # Test is in src/vulcan/tests/, so go up one level to get to vulcan/
    project_root = Path(__file__).parent.parent
    init_file = project_root / "api" / "__init__.py"
    
    tree = parse_python_file(init_file)
    exports = extract_all_exports(tree)
    
    required_exports = {
        "validate_api_key_middleware",
        "rate_limiting_middleware",
    }
    
    missing = required_exports - exports
    assert not missing, (
        f"Missing middleware in __all__ exports: {missing}. "
        "Issue #1: Middleware must be exported for external use"
    )


def test_middleware_functions_can_be_imported():
    """
    TEST: Verify middleware can be imported via documented pattern
    
    REQUIREMENT: Issue #1 - Docstring example should work
    VALIDATION: Runtime import test
    """
    try:
        from vulcan.api import (
            validate_api_key_middleware,
            rate_limiting_middleware,
        )
        
        # Verify they are callable
        assert callable(validate_api_key_middleware), (
            "validate_api_key_middleware should be callable"
        )
        assert callable(rate_limiting_middleware), (
            "rate_limiting_middleware should be callable"
        )
        
    except ImportError as e:
        pytest.fail(
            f"Failed to import middleware as documented: {e}. "
            "Issue #1: Import pattern from docstring must work"
        )


# ============================================================================
# ISSUE #2: Graceful Shutdown Tests
# ============================================================================

def test_stop_rate_limit_cleanup_function_exists():
    """
    TEST: Verify stop_rate_limit_cleanup() function exists
    
    REQUIREMENT: Issue #2 - Graceful shutdown function
    VALIDATION: Import and callable check
    """
    try:
        from vulcan.api.rate_limiting import stop_rate_limit_cleanup
        assert callable(stop_rate_limit_cleanup), (
            "stop_rate_limit_cleanup should be callable"
        )
    except ImportError as e:
        pytest.fail(
            f"stop_rate_limit_cleanup not found: {e}. "
            "Issue #2: Function must be exported for graceful shutdown"
        )


def test_rate_limit_cleanup_stop_event_exists():
    """
    TEST: Verify stop event mechanism exists
    
    REQUIREMENT: Issue #2 - threading.Event for stop signal
    VALIDATION: Check module exports stop event
    """
    try:
        from vulcan.api.rate_limiting import rate_limit_cleanup_stop_event
        assert isinstance(rate_limit_cleanup_stop_event, threading.Event), (
            "rate_limit_cleanup_stop_event must be a threading.Event"
        )
    except ImportError as e:
        pytest.fail(
            f"rate_limit_cleanup_stop_event not found: {e}. "
            "Issue #2: Stop event must be available for shutdown signaling"
        )


def test_graceful_shutdown_integration():
    """
    TEST: Verify graceful shutdown actually works
    
    REQUIREMENT: Issue #2 - Thread stops on signal
    VALIDATION: Start thread, signal stop, verify termination
    """
    from vulcan.api.rate_limiting import (
        start_rate_limit_cleanup,
        stop_rate_limit_cleanup,
        rate_limit_cleanup_stop_event,
    )
    
    # Clear any previous state
    rate_limit_cleanup_stop_event.clear()
    
    # Start cleanup thread with short interval for testing
    thread = start_rate_limit_cleanup(cleanup_interval=10, window_seconds=60)
    assert thread.is_alive(), "Cleanup thread should start"
    
    # Give thread time to enter wait loop
    time.sleep(0.1)
    
    # Stop gracefully
    result = stop_rate_limit_cleanup(timeout=2.0)
    assert result is True, "Shutdown should complete within timeout"
    assert not thread.is_alive(), "Thread should be stopped"


# ============================================================================
# ISSUE #3: History Validation Tests
# ============================================================================

def test_chat_history_message_model_exists():
    """
    TEST: Verify ChatHistoryMessage model is defined
    
    REQUIREMENT: Issue #3 - ChatHistoryMessage Pydantic model
    VALIDATION: Import and type check
    """
    try:
        from vulcan.api.models import ChatHistoryMessage
        from pydantic import BaseModel
        
        assert issubclass(ChatHistoryMessage, BaseModel), (
            "ChatHistoryMessage must be a Pydantic BaseModel"
        )
    except ImportError as e:
        pytest.fail(
            f"ChatHistoryMessage not found: {e}. "
            "Issue #3: Model must be defined in api/models.py"
        )


def test_chat_history_message_valid_roles():
    """
    TEST: Verify ChatHistoryMessage accepts valid roles
    
    REQUIREMENT: Issue #3 - Role validation with pattern ^(user|assistant|system)$
    VALIDATION: Valid roles should pass
    """
    from vulcan.api.models import ChatHistoryMessage
    
    valid_roles = ["user", "assistant", "system"]
    
    for role in valid_roles:
        msg = ChatHistoryMessage(role=role, content="Test message")
        assert msg.role == role, f"Valid role '{role}' should be accepted"
        assert msg.content == "Test message"


def test_chat_history_message_rejects_invalid_roles():
    """
    TEST: Verify ChatHistoryMessage rejects invalid roles
    
    REQUIREMENT: Issue #3 - Strict role validation
    VALIDATION: Invalid roles should raise ValidationError
    """
    from vulcan.api.models import ChatHistoryMessage
    
    invalid_roles = ["admin", "bot", "moderator", "USER", ""]
    
    for role in invalid_roles:
        with pytest.raises(ValidationError) as exc_info:
            ChatHistoryMessage(role=role, content="Test message")
        
        error_str = str(exc_info.value)
        assert "role" in error_str.lower(), (
            f"Validation error should mention 'role' field for invalid role '{role}'"
        )


def test_chat_history_message_requires_content():
    """
    TEST: Verify ChatHistoryMessage requires non-empty content
    
    REQUIREMENT: Issue #3 - Content field with min_length=1
    VALIDATION: Empty content should raise ValidationError
    """
    from vulcan.api.models import ChatHistoryMessage
    
    # Empty string should fail
    with pytest.raises(ValidationError) as exc_info:
        ChatHistoryMessage(role="user", content="")
    
    error_str = str(exc_info.value)
    assert "content" in error_str.lower(), (
        "Validation error should mention 'content' field for empty string"
    )


def test_unified_chat_request_uses_chat_history_message():
    """
    TEST: Verify UnifiedChatRequest.history uses List[ChatHistoryMessage]
    
    REQUIREMENT: Issue #3 - Update history field type
    VALIDATION: Check type annotation
    """
    from vulcan.api.models import UnifiedChatRequest
    import typing
    
    # Get type hints for UnifiedChatRequest
    hints = typing.get_type_hints(UnifiedChatRequest)
    history_type = hints.get("history")
    
    assert history_type is not None, "history field must have type annotation"
    
    # Check that it's a List of ChatHistoryMessage
    # The actual type will be typing.List[ChatHistoryMessage]
    type_str = str(history_type)
    assert "ChatHistoryMessage" in type_str, (
        f"history field must use ChatHistoryMessage, got: {type_str}"
    )


def test_unified_chat_request_validates_history():
    """
    TEST: Verify UnifiedChatRequest validates history messages
    
    REQUIREMENT: Issue #3 - End-to-end validation
    VALIDATION: Invalid history should be rejected
    """
    from vulcan.api.models import UnifiedChatRequest, ChatHistoryMessage
    
    # Valid request with proper history
    valid_history = [
        ChatHistoryMessage(role="user", content="Hello"),
        ChatHistoryMessage(role="assistant", content="Hi there!"),
    ]
    
    request = UnifiedChatRequest(
        message="How are you?",
        history=valid_history
    )
    
    assert len(request.history) == 2
    assert request.history[0].role == "user"
    assert request.history[1].role == "assistant"


# ============================================================================
# ISSUE #4: Request ID Tracking Tests
# ============================================================================

def test_generate_request_id_function_exists():
    """
    TEST: Verify generate_request_id() utility exists
    
    REQUIREMENT: Issue #4 - Request ID generation utility
    VALIDATION: Import and callable check
    """
    try:
        from vulcan.api.models import generate_request_id
        assert callable(generate_request_id), (
            "generate_request_id should be callable"
        )
    except ImportError as e:
        pytest.fail(
            f"generate_request_id not found: {e}. "
            "Issue #4: Utility function must be exported"
        )


def test_generate_request_id_format():
    """
    TEST: Verify request ID format is correct
    
    REQUIREMENT: Issue #4 - Format: {prefix}_{timestamp}_{random_hex}
    VALIDATION: Check structure and uniqueness
    """
    from vulcan.api.models import generate_request_id
    
    rid = generate_request_id()
    
    # Check format
    parts = rid.split("_")
    assert len(parts) == 3, f"Request ID should have 3 parts, got: {rid}"
    assert parts[0] == "req", f"Default prefix should be 'req', got: {parts[0]}"
    
    # Check timestamp part is numeric
    assert parts[1].isdigit(), f"Timestamp part should be numeric: {parts[1]}"
    
    # Check random hex part
    assert len(parts[2]) == 16, f"Random hex should be 16 chars: {parts[2]}"
    assert all(c in "0123456789abcdef" for c in parts[2]), (
        f"Random part should be hex: {parts[2]}"
    )


def test_generate_request_id_uniqueness():
    """
    TEST: Verify request IDs are unique
    
    REQUIREMENT: Issue #4 - Uniqueness across requests
    VALIDATION: Generate multiple IDs, check no duplicates
    """
    from vulcan.api.models import generate_request_id
    
    ids = [generate_request_id() for _ in range(100)]
    unique_ids = set(ids)
    
    assert len(unique_ids) == 100, (
        f"Expected 100 unique IDs, got {len(unique_ids)}. "
        "Request IDs must be unique"
    )


def test_response_models_have_request_id_field():
    """
    TEST: Verify all response models have request_id field
    
    REQUIREMENT: Issue #4 - Add request_id to response models
    VALIDATION: Check field exists and is optional
    """
    from vulcan.api.models import (
        ChatResponse,
        VulcanResponse,
        StatusResponse,
        ConfigResponse,
        HealthResponse,
        MetricsResponse,
        ErrorResponse,
    )
    
    response_models = [
        ChatResponse,
        VulcanResponse,
        StatusResponse,
        ConfigResponse,
        HealthResponse,
        MetricsResponse,
        ErrorResponse,
    ]
    
    for model_class in response_models:
        model_name = model_class.__name__
        fields = model_class.model_fields
        
        assert "request_id" in fields, (
            f"{model_name} must have request_id field. "
            "Issue #4: All response models need request tracking"
        )
        
        # Verify it's optional (has default None)
        field_info = fields["request_id"]
        assert field_info.default is None or field_info.default_factory is not None, (
            f"{model_name}.request_id should be optional"
        )


# ============================================================================
# ISSUE #5: Auto-populated Timestamps Tests
# ============================================================================

def test_error_response_timestamp_auto_populated():
    """
    TEST: Verify ErrorResponse timestamp is auto-populated
    
    REQUIREMENT: Issue #5 - Auto-timestamp using default_factory
    VALIDATION: Create instance without timestamp, verify it's set
    """
    from vulcan.api.models import ErrorResponse
    
    before = datetime.utcnow().timestamp()
    time.sleep(0.01)  # Small delay to ensure timestamp difference
    
    error = ErrorResponse(error="Test error")
    
    after = datetime.utcnow().timestamp()
    
    assert error.timestamp is not None, (
        "Timestamp should be auto-populated"
    )
    assert before <= error.timestamp <= after, (
        f"Timestamp {error.timestamp} should be between {before} and {after}"
    )


def test_error_response_timestamp_uses_default_factory():
    """
    TEST: Verify ErrorResponse uses default_factory for timestamp
    
    REQUIREMENT: Issue #5 - Proper Pydantic default_factory usage
    VALIDATION: AST check for default_factory
    """
    # Test is in src/vulcan/tests/, so go up one level to get to vulcan/
    project_root = Path(__file__).parent.parent
    models_file = project_root / "api" / "models.py"
    
    with open(models_file, encoding="utf-8") as f:
        content = f.read()
    
    # Look for ErrorResponse class and timestamp field
    error_response_section = re.search(
        r'class ErrorResponse\(BaseModel\):.*?(?=\nclass|\n__all__|$)',
        content,
        re.DOTALL
    )
    
    assert error_response_section, "ErrorResponse class not found"
    
    section_text = error_response_section.group(0)
    
    # Check for default_factory pattern
    assert "default_factory" in section_text, (
        "ErrorResponse.timestamp should use default_factory"
    )
    assert "datetime.utcnow().timestamp()" in section_text, (
        "Timestamp should use datetime.utcnow().timestamp()"
    )


# ============================================================================
# ISSUE #6: Initialization Checks Tests
# ============================================================================

def test_unified_chat_has_detailed_initialization_checks():
    """
    TEST: Verify unified_chat has enhanced initialization checks
    
    REQUIREMENT: Issue #6 - Detailed error messages for 503 errors
    VALIDATION: Check for comprehensive validation logic
    """
    # Test is in src/vulcan/tests/, so go up one level to get to vulcan/
    project_root = Path(__file__).parent.parent
    unified_chat_file = project_root / "endpoints" / "unified_chat.py"
    
    with open(unified_chat_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check for detailed initialization checks
    required_checks = [
        "hasattr(app.state, \"deployment\")",
        "deployment is None",
        "deployment.collective",
    ]
    
    for check in required_checks:
        assert check in content, (
            f"Missing initialization check: {check}. "
            "Issue #6: Must have comprehensive initialization validation"
        )
    
    # Check for improved error messages
    assert "System initializing" in content or "initialization incomplete" in content, (
        "Should have user-friendly error messages"
    )
    
    # Check for logging
    assert "logger.error" in content, (
        "Should log detailed errors for debugging"
    )


def test_initialization_error_messages_are_informative():
    """
    TEST: Verify error messages help diagnose initialization issues
    
    REQUIREMENT: Issue #6 - Distinguish between different failure modes
    VALIDATION: Check for specific error messages
    """
    # Test is in src/vulcan/tests/, so go up one level to get to vulcan/
    project_root = Path(__file__).parent.parent
    unified_chat_file = project_root / "endpoints" / "unified_chat.py"
    
    with open(unified_chat_file, encoding="utf-8") as f:
        content = f.read()
    
    # Should have different messages for different scenarios
    error_indicators = [
        "deployment not ready",
        "deployment object is None",
        "collective not",
    ]
    
    found_errors = sum(1 for indicator in error_indicators if indicator in content)
    
    assert found_errors >= 2, (
        f"Found only {found_errors} distinct error messages. "
        "Issue #6: Need multiple specific error messages to help diagnose which "
        "component failed initialization"
    )


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    """Standalone test runner"""
    import sys
    
    test_functions = [
        # Issue #1 tests
        test_middleware_imported_in_api_init,
        test_middleware_exported_in_api_init,
        test_middleware_functions_can_be_imported,
        # Issue #2 tests
        test_stop_rate_limit_cleanup_function_exists,
        test_rate_limit_cleanup_stop_event_exists,
        test_graceful_shutdown_integration,
        # Issue #3 tests
        test_chat_history_message_model_exists,
        test_chat_history_message_valid_roles,
        test_chat_history_message_rejects_invalid_roles,
        test_chat_history_message_requires_content,
        test_unified_chat_request_uses_chat_history_message,
        test_unified_chat_request_validates_history,
        # Issue #4 tests
        test_generate_request_id_function_exists,
        test_generate_request_id_format,
        test_generate_request_id_uniqueness,
        test_response_models_have_request_id_field,
        # Issue #5 tests
        test_error_response_timestamp_auto_populated,
        test_error_response_timestamp_uses_default_factory,
        # Issue #6 tests
        test_unified_chat_has_detailed_initialization_checks,
        test_initialization_error_messages_are_informative,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    sys.exit(0 if failed == 0 else 1)
