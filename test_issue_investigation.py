"""
Investigation script for Issue #1: Math Tool Computes Wrong Formula

This script tests the current behavior of the mathematical computation tool
to understand the root cause of the bug where Σ(2k-1) is computed as Σk.

Expected: Σ(2k-1) from k=1 to n should equal n²
Actual: Returns n*(n+1)/2 (which is Σk, not Σ(2k-1))
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

try:
    from vulcan.reasoning.mathematical_computation import (
        MathematicalComputationTool,
        CodeTemplates,
        create_mathematical_computation_tool,
    )
    TOOL_AVAILABLE = True
except ImportError as e:
    print(f"Cannot import mathematical computation tool: {e}")
    TOOL_AVAILABLE = False
    sys.exit(1)

def test_summation_extraction():
    """Test if the summation expression is correctly extracted"""
    print("\n" + "="*80)
    print("TEST 1: Expression Extraction")
    print("="*80)
    
    templates = CodeTemplates()
    
    # Test query
    query = "∑(2k-1) from k=1 to n"
    print(f"\nQuery: {query}")
    
    # This should extract and return code for Σ(2k-1)
    code = templates.generate_from_query(query)
    print(f"\nGenerated code:\n{code}")
    
    # Check if the expression "(2k-1)" or "2*k-1" appears in the code
    if "2*k" in code or "2k" in code:
        print("\n✓ Expression extraction looks correct - contains '2k'")
    else:
        print("\n✗ PROBLEM: Expression '2k' not found in generated code!")
        print("   This means the summand is not being correctly extracted.")
    
    return code

def test_summation_computation():
    """Test if the computation produces the correct result"""
    print("\n" + "="*80)
    print("TEST 2: Computation Result")
    print("="*80)
    
    try:
        tool = create_mathematical_computation_tool()
        
        # Test with the problematic query
        query = "Compute ∑(2k-1) from k=1 to n"
        print(f"\nQuery: {query}")
        
        result = tool.execute(query)
        
        print(f"\nResult type: {type(result)}")
        print(f"Result: {result}")
        
        if hasattr(result, 'result'):
            print(f"\nComputed result: {result.result}")
            
            # Check if the result is n**2 (correct) or n*(n+1)/2 (incorrect)
            result_str = str(result.result)
            
            if 'n**2' in result_str or 'n^2' in result_str:
                print("\n✓ CORRECT: Result is n²")
            elif 'n*(n+1)/2' in result_str or 'n(n+1)/2' in result_str:
                print("\n✗ BUG CONFIRMED: Result is n*(n+1)/2 instead of n²")
                print("   This is the formula for Σk, not Σ(2k-1)")
            else:
                print(f"\n? UNEXPECTED: Result is {result_str}")
                
        if hasattr(result, 'code'):
            print(f"\nGenerated code:\n{result.code}")
            
    except Exception as e:
        print(f"\n✗ ERROR during computation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if TOOL_AVAILABLE:
        print("\nInvestigating Issue #1: Math Tool Computes Wrong Formula")
        print("="*80)
        
        # Run tests
        test_summation_extraction()
        test_summation_computation()
        
        print("\n" + "="*80)
        print("Investigation complete")
        print("="*80)
    else:
        print("Tool not available for testing")
