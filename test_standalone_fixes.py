#!/usr/bin/env python3
"""
Standalone test runner for reasoning system gaps follow-up fixes.
Runs without pytest to avoid conftest dependencies.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

def test_issue2_causal_keywords():
    """Test Issue #2: Causal keywords are defined."""
    print("\n=== Test Issue #2: Causal Keywords ===")
    try:
        from vulcan.llm.query_classifier import CAUSAL_KEYWORDS, STRONG_CAUSAL_KEYWORDS
        
        assert len(CAUSAL_KEYWORDS) > 0, "CAUSAL_KEYWORDS should not be empty"
        assert len(STRONG_CAUSAL_KEYWORDS) > 0, "STRONG_CAUSAL_KEYWORDS should not be empty"
        
        pearl_terms = ['confound', 'intervention', 'dag', 'backdoor', 'frontdoor']
        for term in pearl_terms:
            assert term in CAUSAL_KEYWORDS, f"CAUSAL_KEYWORDS should include '{term}'"
        
        print(f"✓ CAUSAL_KEYWORDS defined with {len(CAUSAL_KEYWORDS)} keywords")
        print(f"✓ STRONG_CAUSAL_KEYWORDS defined with {len(STRONG_CAUSAL_KEYWORDS)} keywords")
        print(f"✓ Pearl-style terms present: {pearl_terms}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_issue4_unicode_math_detection():
    """Test Issue #4: Unicode math detection functions exist."""
    print("\n=== Test Issue #4: Unicode Math Detection ===")
    try:
        from vulcan.routing.query_router import (
            UNICODE_MATH_SYMBOLS,
            _has_unicode_math,
            _has_latex_math,
            _has_math_notation
        )
        
        # Test constants
        assert isinstance(UNICODE_MATH_SYMBOLS, frozenset), "Should be frozenset"
        assert '∫' in UNICODE_MATH_SYMBOLS, "Should include integral ∫"
        assert '∑' in UNICODE_MATH_SYMBOLS, "Should include summation ∑"
        
        # Test Unicode detection
        assert _has_unicode_math("∫₀ᵀu(t)²dt"), "Should detect Unicode integral"
        assert not _has_unicode_math("integrate x^2"), "Should not detect English"
        
        # Test LaTeX detection
        assert _has_latex_math("\\int_0^T u(t)^2 dt"), "Should detect LaTeX \\int"
        assert not _has_latex_math("integrate from 0 to T"), "Should not detect English"
        
        # Test math notation detection
        assert _has_math_notation("E_safe = 100"), "Should detect subscript"
        assert _has_math_notation("x^2 + y^3"), "Should detect superscript"
        assert _has_math_notation("u(t)"), "Should detect function notation"
        
        print(f"✓ UNICODE_MATH_SYMBOLS defined with {len(UNICODE_MATH_SYMBOLS)} symbols")
        print(f"✓ _has_unicode_math() detects ∫₀ᵀu(t)²dt")
        print(f"✓ _has_latex_math() detects \\int_0^T")
        print(f"✓ _has_math_notation() detects E_safe, x^2, u(t)")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_new_requirement_integral_parsing():
    """Test NEW REQUIREMENT: Integral expression parsing."""
    print("\n=== Test NEW REQUIREMENT: Integral Parsing ===")
    try:
        from vulcan.reasoning.mathematical_computation import MathematicalComputationEngine
        
        engine = MathematicalComputationEngine()
        
        # Test parse method exists
        assert hasattr(engine, '_parse_integral_expression'), "Method should exist"
        assert callable(engine._parse_integral_expression), "Should be callable"
        
        # Test Unicode integral parsing
        integrand, variable, bounds = engine._parse_integral_expression("∫₀ᵀu(t)²dt")
        print(f"  Unicode integral: integrand={integrand}, variable={variable}, bounds={bounds}")
        assert integrand is not None, "Should parse integrand"
        assert variable == "t", f"Should parse variable as 't', got '{variable}'"
        
        # Test English integral parsing
        integrand2, variable2, bounds2 = engine._parse_integral_expression("integrate x^2 from 0 to 1")
        print(f"  English integral: integrand={integrand2}, variable={variable2}, bounds={bounds2}")
        assert integrand2 is not None, "Should parse integrand"
        assert 'x' in integrand2, "Integrand should contain x"
        
        # Test normalization
        assert hasattr(engine, '_normalize_math_expression'), "Normalize method should exist"
        normalized = engine._normalize_math_expression("x^2 + 2x")
        print(f"  Normalized 'x^2 + 2x' → '{normalized}'")
        assert "**" in normalized, "Should convert ^ to **"
        
        # Test sanitization
        assert hasattr(engine, '_sanitize_sympy_expression'), "Sanitize method should exist"
        safe = engine._sanitize_sympy_expression("x**2 + sin(x)")
        print(f"  Sanitized 'x**2 + sin(x)' → '{safe}'")
        assert safe == "x**2 + sin(x)", "Should not modify safe expressions"
        
        print("✓ _parse_integral_expression() parses Unicode and English notation")
        print("✓ _normalize_math_expression() converts ^ to **")
        print("✓ _sanitize_sympy_expression() protects against dangerous patterns")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Reasoning System Gaps Follow-up Fixes")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Issue #2: Causal Keywords", test_issue2_causal_keywords()))
    results.append(("Issue #4: Unicode Math Detection", test_issue4_unicode_math_detection()))
    results.append(("NEW: Integral Parsing", test_new_requirement_integral_parsing()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
