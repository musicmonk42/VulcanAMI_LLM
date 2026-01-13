#!/usr/bin/env python3
"""
Test for the _should_use_cross_domain_transfer method signature fix.

This test validates the fix for the TypeError that was occurring in the
reasoning integration pipeline:

TypeError: ReasoningIntegration._should_use_cross_domain_transfer() takes 
2 positional arguments but 3 were given

The fix ensures that the method is called with the correct number of arguments
matching its signature.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_should_use_cross_domain_transfer_method_signature():
    """
    Verify that _should_use_cross_domain_transfer has the correct signature
    and can be called with only selected_tools argument.
    """
    from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
    import inspect
    
    # Create ReasoningIntegration instance
    integration = ReasoningIntegration()
    
    # Check method exists
    assert hasattr(integration, '_should_use_cross_domain_transfer'), \
        "ReasoningIntegration must have _should_use_cross_domain_transfer method"
    
    # Check method signature
    sig = inspect.signature(integration._should_use_cross_domain_transfer)
    params = list(sig.parameters.keys())
    
    # Method should only have 'selected_tools' parameter (self is implicit)
    assert params == ['selected_tools'], \
        f"Method should only have 'selected_tools' parameter, got {params}"
    
    print("✓ Method signature is correct: only 'selected_tools' parameter")
    
    # Test that method can be called with just selected_tools
    try:
        result = integration._should_use_cross_domain_transfer(['tool1', 'tool2'])
        print(f"✓ Method call succeeded with result: {result}")
    except TypeError as e:
        raise AssertionError(f"Method call failed with TypeError: {e}")
    
    # Test that calling with extra arguments raises TypeError
    try:
        # This should fail - method doesn't accept query_analysis
        result = integration._should_use_cross_domain_transfer(
            ['tool1', 'tool2'],
            {"type": "general", "complexity": 0.5}
        )
        raise AssertionError(
            "Method should have raised TypeError when called with extra arguments"
        )
    except TypeError as e:
        # This is expected
        print(f"✓ Method correctly rejects extra arguments: {e}")
    
    print("\n✓ All tests passed!")


def test_apply_reasoning_calls_method_correctly():
    """
    Verify that apply_reasoning_impl.py calls _should_use_cross_domain_transfer
    with the correct number of arguments.
    """
    import ast
    import os
    
    # Read the apply_reasoning_impl.py file
    file_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'src', 
        'vulcan', 
        'reasoning', 
        'integration', 
        'apply_reasoning_impl.py'
    )
    
    with open(file_path, 'r') as f:
        source = f.read()
    
    # Parse the source code
    tree = ast.parse(source)
    
    # Find all calls to _should_use_cross_domain_transfer
    calls_found = []
    
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == '_should_use_cross_domain_transfer':
                    calls_found.append({
                        'line': node.lineno,
                        'num_args': len(node.args),
                        'num_keywords': len(node.keywords)
                    })
            self.generic_visit(node)
    
    visitor = CallVisitor()
    visitor.visit(tree)
    
    # Verify we found at least one call
    assert len(calls_found) > 0, \
        "Should have found at least one call to _should_use_cross_domain_transfer"
    
    # Verify each call has exactly 1 argument (selected_tools)
    for call in calls_found:
        assert call['num_args'] == 1, \
            f"Call at line {call['line']} has {call['num_args']} arguments, expected 1"
        print(f"✓ Call at line {call['line']} has correct number of arguments (1)")
    
    print("\n✓ All method calls are correct!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing _should_use_cross_domain_transfer signature fix")
    print("=" * 70)
    print()
    
    print("Test 1: Method signature validation")
    print("-" * 70)
    test_should_use_cross_domain_transfer_method_signature()
    
    print()
    print("Test 2: Apply reasoning calls method correctly")
    print("-" * 70)
    test_apply_reasoning_calls_method_correctly()
    
    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
