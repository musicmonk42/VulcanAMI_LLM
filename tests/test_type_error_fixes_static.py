"""
Static Test for Type Error Fixes

This test validates the fixes for two production errors using static analysis:
1. unified_chat.py: AttributeError when direct_conclusion is dict instead of string
2. apply_reasoning_impl.py: Missing _is_self_referential attribute in ReasoningIntegration

Uses AST parsing to verify fixes without requiring runtime imports.
"""

import ast
import inspect
from pathlib import Path


class TestUnifiedChatFixStatic:
    """Static analysis tests for unified_chat.py fix."""
    
    def test_normalize_conclusion_function_exists(self):
        """Test that _normalize_conclusion_to_string function exists."""
        unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
        assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
        
        with open(unified_chat_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(unified_chat_file))
        
        # Find the function
        func_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_normalize_conclusion_to_string":
                func_found = True
                # Check it has correct parameter
                assert len(node.args.args) >= 1, "Function should have at least 1 parameter"
                assert node.args.args[0].arg == "conclusion", "First parameter should be 'conclusion'"
                break
        
        assert func_found, "_normalize_conclusion_to_string function not found"
        print("✓ _normalize_conclusion_to_string function exists with correct signature")
    
    def test_normalize_conclusion_called_before_strip(self):
        """Test that conclusions are normalized before calling strip()."""
        unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
        
        with open(unified_chat_file, 'r') as f:
            content = f.read()
        
        # Check that _normalize_conclusion_to_string is called on conclusions
        assert "_normalize_conclusion_to_string(unified_conclusion)" in content, \
            "unified_conclusion should be normalized"
        assert "_normalize_conclusion_to_string(agent_conclusion)" in content, \
            "agent_conclusion should be normalized"
        assert "_normalize_conclusion_to_string(direct_conclusion)" in content, \
            "direct_conclusion should be normalized"
        
        print("✓ All conclusions are normalized before use")
    
    def test_type_check_before_strip(self):
        """Test that isinstance check is done before calling strip()."""
        unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
        
        with open(unified_chat_file, 'r') as f:
            content = f.read()
        
        # Find the section with the strip() calls
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'direct_conclusion' in line and '.strip()' in line:
                # Check that there's an isinstance check nearby
                context = '\n'.join(lines[max(0, i-5):i+5])
                assert 'isinstance' in context and 'str' in context, \
                    "isinstance check should be present near strip() call"
                print(f"✓ Type check found near line {i+1}")
                break


class TestReasoningIntegrationFixStatic:
    """Static analysis tests for ReasoningIntegration fix."""
    
    def test_wrapper_methods_exist(self):
        """Test that wrapper methods exist in ReasoningIntegration class."""
        orchestrator_file = Path("src/vulcan/reasoning/integration/orchestrator.py")
        assert orchestrator_file.exists(), f"File not found: {orchestrator_file}"
        
        with open(orchestrator_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(orchestrator_file))
        
        # Find the ReasoningIntegration class
        reasoning_integration_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ReasoningIntegration":
                reasoning_integration_class = node
                break
        
        assert reasoning_integration_class is not None, "ReasoningIntegration class not found"
        
        # Get all method names
        method_names = [
            item.name for item in reasoning_integration_class.body
            if isinstance(item, ast.FunctionDef)
        ]
        
        # Check for required wrapper methods
        required_methods = [
            '_is_self_referential',
            '_is_ethical_query',
            '_consult_world_model_introspection'
        ]
        
        for method in required_methods:
            assert method in method_names, f"Method {method} not found in ReasoningIntegration class"
            print(f"✓ Method {method} exists in ReasoningIntegration")
    
    def test_imports_from_query_analysis(self):
        """Test that required functions are imported from query_analysis."""
        orchestrator_file = Path("src/vulcan/reasoning/integration/orchestrator.py")
        
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for imports
        assert "from .query_analysis import" in content, \
            "Should import from query_analysis module"
        assert "is_self_referential" in content, \
            "Should import is_self_referential"
        assert "is_ethical_query" in content, \
            "Should import is_ethical_query"
        assert "consult_world_model_introspection" in content, \
            "Should import consult_world_model_introspection"
        
        print("✓ All required functions are imported from query_analysis")
    
    def test_wrapper_methods_delegate_to_imports(self):
        """Test that wrapper methods delegate to imported functions."""
        orchestrator_file = Path("src/vulcan/reasoning/integration/orchestrator.py")
        
        with open(orchestrator_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(orchestrator_file))
        
        # Find the ReasoningIntegration class
        reasoning_integration_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ReasoningIntegration":
                reasoning_integration_class = node
                break
        
        # Check _is_self_referential method
        for item in reasoning_integration_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_is_self_referential":
                # Should have a return statement that calls is_self_referential
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Return) and stmt.value:
                        if isinstance(stmt.value, ast.Call):
                            if hasattr(stmt.value.func, 'id'):
                                assert stmt.value.func.id == "is_self_referential", \
                                    "_is_self_referential should call is_self_referential"
                                print("✓ _is_self_referential delegates correctly")


def run_all_tests():
    """Run all static tests."""
    test_classes = [TestUnifiedChatFixStatic, TestReasoningIntegrationFixStatic]
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                print(f"\n{method_name}:")
                method = getattr(test_instance, method_name)
                method()
                print(f"✅ PASSED")
            except AssertionError as e:
                print(f"❌ FAILED: {e}")
                return False
            except Exception as e:
                print(f"💥 ERROR: {e}")
                return False
    
    print(f"\n{'='*60}")
    print("🎉 All tests passed!")
    print('='*60)
    return True


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
