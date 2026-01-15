#!/usr/bin/env python3
"""
Test suite for CRITICAL FIX #2: FOL Quantifier Scope Ambiguity Handler.

Tests the _handle_fol_formalization() method with various inputs to ensure:
1. Correct detection of quantifier scope ambiguity
2. Generation of both readings (narrow and wide scope)
3. Proper FOL formalization with correct syntax
4. Appropriate confidence scores
5. Fallback handling for non-ambiguous cases
"""

import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner


def test_quantifier_scope_ambiguity():
    """Test the core use case: quantifier scope ambiguity detection."""
    print("\n" + "="*80)
    print("TEST 1: Quantifier Scope Ambiguity - 'Every engineer reviewed a document'")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    query = """Language Reasoning - Formalize quantifier scope ambiguity in FOL:
Sentence: "Every engineer reviewed a document."
"""
    
    result = reasoner._handle_fol_formalization(query)
    
    print("\n📋 Input Query:")
    print(query)
    
    print("\n✅ Result:")
    print(json.dumps(result, indent=2))
    
    # Assertions
    assert result["proven"] == True, "Should be proven"
    assert result["confidence"] == 0.90, f"Confidence should be 0.90, got {result['confidence']}"
    assert result["applicable"] == True, "Should be applicable"
    assert result["method"] == "fol_formalization", "Method should be fol_formalization"
    
    fol_form = result["fol_formalization"]
    assert "original_sentence" in fol_form, "Should have original_sentence"
    assert "reading_a" in fol_form, "Should have reading_a"
    assert "reading_b" in fol_form, "Should have reading_b"
    assert "ambiguity_type" in fol_form, "Should have ambiguity_type"
    assert fol_form["ambiguity_type"] == "quantifier_scope", "Should detect quantifier_scope"
    
    # Check Reading A (narrow scope - one shared document)
    reading_a = fol_form["reading_a"]
    assert "∃d.(∀e." in reading_a["fol"], f"Reading A should have ∃d.(∀e. pattern, got {reading_a['fol']}"
    assert "Narrow scope" in reading_a["interpretation"], "Reading A should indicate narrow scope"
    assert "one shared" in reading_a["interpretation"].lower(), "Reading A should mention shared object"
    
    # Check Reading B (wide scope - possibly different documents)
    reading_b = fol_form["reading_b"]
    assert "∀e.(∃d." in reading_b["fol"], f"Reading B should have ∀e.(∃d. pattern, got {reading_b['fol']}"
    assert "Wide scope" in reading_b["interpretation"], "Reading B should indicate wide scope"
    assert "possibly different" in reading_b["interpretation"].lower(), "Reading B should mention different objects"
    
    print("\n✅ TEST 1 PASSED!")
    return True


def test_alternate_patterns():
    """Test alternate quantifier patterns: 'each', 'all', 'some'."""
    print("\n" + "="*80)
    print("TEST 2: Alternate Patterns - 'Each student read some book'")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    test_cases = [
        ("Each student read some book.", "student", "book", "s", "b"),
        ("All teachers assigned a project.", "teacher", "project", "t", "p"),
        ("Every developer tested a feature.", "developer", "feature", "d", "f"),
    ]
    
    for sentence, subject, obj, subj_var, obj_var in test_cases:
        print(f"\n📋 Testing: {sentence}")
        
        query = f'Formalize: "{sentence}"'
        result = reasoner._handle_fol_formalization(query)
        
        assert result["confidence"] == 0.90, f"Should have high confidence for: {sentence}"
        fol_form = result["fol_formalization"]
        
        # Verify both readings exist
        assert "reading_a" in fol_form, f"Missing reading_a for: {sentence}"
        assert "reading_b" in fol_form, f"Missing reading_b for: {sentence}"
        
        # Verify structure
        reading_a_fol = fol_form["reading_a"]["fol"]
        reading_b_fol = fol_form["reading_b"]["fol"]
        
        print(f"  Reading A: {reading_a_fol}")
        print(f"  Reading B: {reading_b_fol}")
        
        assert "∃" in reading_a_fol and "∀" in reading_a_fol, f"Reading A should have both quantifiers: {reading_a_fol}"
        assert "∀" in reading_b_fol and "∃" in reading_b_fol, f"Reading B should have both quantifiers: {reading_b_fol}"
        
        print(f"  ✅ Passed")
    
    print("\n✅ TEST 2 PASSED!")
    return True


def test_sentence_extraction():
    """Test various sentence extraction formats."""
    print("\n" + "="*80)
    print("TEST 3: Sentence Extraction from Various Formats")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    test_cases = [
        ('Sentence: "Every engineer reviewed a document."', "Every engineer reviewed a document."),
        ('Formalize: "Each student read a book."', "Each student read a book."),
        ('FOL for "All teachers assigned a project."', "All teachers assigned a project."),
        ('Sentence: Every developer tested a feature.', "Every developer tested a feature."),
    ]
    
    for query, expected_sentence in test_cases:
        print(f"\n📋 Query: {query}")
        
        extracted = reasoner._extract_sentence_from_query(query)
        print(f"  Extracted: {extracted}")
        print(f"  Expected: {expected_sentence}")
        
        # Normalize for comparison (remove trailing periods)
        extracted_norm = extracted.rstrip('.') if extracted else None
        expected_norm = expected_sentence.rstrip('.')
        
        assert extracted_norm == expected_norm, f"Failed to extract correctly. Got: {extracted}, Expected: {expected_sentence}"
        print(f"  ✅ Passed")
    
    print("\n✅ TEST 3 PASSED!")
    return True


def test_fallback_behavior():
    """Test fallback behavior for non-ambiguous cases."""
    print("\n" + "="*80)
    print("TEST 4: Fallback Behavior for Non-Ambiguous Cases")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    # Cases without quantifier scope ambiguity
    test_cases = [
        "Formalize: The sky is blue.",
        "Formalize: Socrates is mortal.",
        "Convert to FOL: All birds can fly.",  # No 'a/some' in object
    ]
    
    for query in test_cases:
        print(f"\n📋 Query: {query}")
        
        result = reasoner._handle_fol_formalization(query)
        
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Method: {result.get('method')}")
        
        # Fallback should return confidence of 0.70
        assert result["confidence"] <= 0.70, f"Fallback should have lower confidence, got {result['confidence']}"
        assert result["applicable"] == True, "Should still be applicable"
        
        print(f"  ✅ Passed (used fallback)")
    
    print("\n✅ TEST 4 PASSED!")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases and Error Handling")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    test_cases = [
        ("", "Empty query"),
        ("Formalize this.", "No sentence found"),
        ("Random text without structure", "No clear sentence"),
    ]
    
    for query, description in test_cases:
        print(f"\n📋 Testing: {description}")
        print(f"  Query: '{query}'")
        
        try:
            result = reasoner._handle_fol_formalization(query)
            print(f"  Result confidence: {result.get('confidence')}")
            
            # Should not crash and should return a valid result
            assert "confidence" in result, "Should have confidence"
            assert "applicable" in result, "Should have applicable"
            
            print(f"  ✅ Passed (handled gracefully)")
        except Exception as e:
            print(f"  ❌ Failed with exception: {e}")
            raise
    
    print("\n✅ TEST 5 PASSED!")
    return True


def test_expected_output_format():
    """Verify the output matches the exact expected format from requirements."""
    print("\n" + "="*80)
    print("TEST 6: Verify Expected Output Format")
    print("="*80)
    
    reasoner = SymbolicReasoner()
    
    query = """Language Reasoning - Formalize quantifier scope ambiguity in FOL:
Sentence: "Every engineer reviewed a document."
"""
    
    result = reasoner._handle_fol_formalization(query)
    
    # Expected structure from requirements
    expected_keys = ["proven", "confidence", "fol_formalization", "applicable", "method"]
    for key in expected_keys:
        assert key in result, f"Missing required key: {key}"
    
    fol_form = result["fol_formalization"]
    expected_fol_keys = ["original_sentence", "reading_a", "reading_b", "ambiguity_type"]
    for key in expected_fol_keys:
        assert key in fol_form, f"Missing required FOL key: {key}"
    
    reading_keys = ["fol", "interpretation", "english_rewrite"]
    for key in reading_keys:
        assert key in fol_form["reading_a"], f"Missing reading_a key: {key}"
        assert key in fol_form["reading_b"], f"Missing reading_b key: {key}"
    
    print("\n📋 Full Result Structure:")
    print(json.dumps(result, indent=2))
    
    print("\n✅ TEST 6 PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CRITICAL FIX #2: FOL Quantifier Scope Ambiguity - Test Suite")
    print("="*80)
    
    tests = [
        test_quantifier_scope_ambiguity,
        test_alternate_patterns,
        test_sentence_extraction,
        test_fallback_behavior,
        test_edge_cases,
        test_expected_output_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"❌ {failed} tests FAILED")
    else:
        print("✅ ALL TESTS PASSED!")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
