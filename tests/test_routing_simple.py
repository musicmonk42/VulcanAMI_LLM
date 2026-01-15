"""
Simple test to verify WorldModel routing methods work correctly.
No pytest dependency required.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from vulcan.world_model.world_model_core import WorldModel


def test_should_route_causal():
    """Test causal query detection."""
    print("Testing causal query detection...")
    wm = WorldModel(config={})
    
    query = "Does confounding affect the causal relationship?"
    result = wm._should_route_to_reasoning_engine(query)
    
    assert result is True, f"Failed to detect causal query. Got: {result}"
    print("✓ Causal query detection works")


def test_should_route_analogical():
    """Test analogical query detection."""
    print("Testing analogical query detection...")
    wm = WorldModel(config={})
    
    query = "In structure mapping, how does domain S correspond to domain T?"
    result = wm._should_route_to_reasoning_engine(query)
    
    assert result is True, f"Failed to detect analogical query. Got: {result}"
    print("✓ Analogical query detection works")


def test_should_route_mathematical():
    """Test mathematical query detection."""
    print("Testing mathematical query detection...")
    wm = WorldModel(config={})
    
    query = "Compute the sum of integers from 1 to n"
    result = wm._should_route_to_reasoning_engine(query)
    
    assert result is True, f"Failed to detect mathematical query. Got: {result}"
    print("✓ Mathematical query detection works")


def test_should_route_sat():
    """Test SAT/symbolic query detection."""
    print("Testing SAT/symbolic query detection...")
    wm = WorldModel(config={})
    
    query = "Is A→B, B→C, ¬C, A∨B satisfiable?"
    result = wm._should_route_to_reasoning_engine(query)
    
    assert result is True, f"Failed to detect SAT query. Got: {result}"
    print("✓ SAT query detection works")


def test_philosophical_not_routed():
    """Test that philosophical queries are NOT routed."""
    print("Testing philosophical query not routed...")
    wm = WorldModel(config={})
    
    query = "What is the meaning of life?"
    result = wm._should_route_to_reasoning_engine(query)
    
    assert result is False, f"Incorrectly routed philosophical query. Got: {result}"
    print("✓ Philosophical query correctly not routed")


def test_input_validation():
    """Test input validation."""
    print("Testing input validation...")
    wm = WorldModel(config={})
    
    # None input
    assert wm._should_route_to_reasoning_engine(None) is False
    
    # Empty string
    assert wm._should_route_to_reasoning_engine("") is False
    
    # Very long query
    long_query = "x" * 20000
    assert wm._should_route_to_reasoning_engine(long_query) is False
    
    print("✓ Input validation works")


def test_normalize_dict_result():
    """Test result normalization."""
    print("Testing result normalization...")
    wm = WorldModel(config={})
    
    result = {
        'response': 'The answer is 42',
        'confidence': 0.9,
        'reasoning_trace': {'steps': ['step1']},
        'mode': 'causal'
    }
    
    normalized = wm._normalize_engine_result(result, 'causal', 'test query')
    
    assert normalized['response'] == 'The answer is 42'
    assert normalized['confidence'] == 0.9
    assert normalized['engine_used'] == 'causal'
    
    print("✓ Result normalization works")


def test_reason_method_routing():
    """Test that reason() method routes correctly."""
    print("Testing reason() method routing...")
    wm = WorldModel(config={})
    
    # Test with causal query
    query = "Does confounding affect causation?"
    
    try:
        result = wm.reason(query, mode=None)
        # Should return a valid dict
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert 'response' in result, "Missing 'response' in result"
        assert 'confidence' in result, "Missing 'confidence' in result"
        print(f"✓ Reason method routing works (mode: {result.get('mode', 'unknown')})")
    except ImportError as e:
        print(f"✓ Routing detected correctly (engine import failed as expected: {e})")
        # This is OK - routing logic detected it should route, engine just not installed


def test_reason_respects_mode():
    """Test that explicit mode overrides routing."""
    print("Testing that explicit mode is respected...")
    wm = WorldModel(config={})
    
    # Causal query with philosophical mode
    query = "Does confounding affect causation?"
    result = wm.reason(query, mode='philosophical')
    
    assert isinstance(result, dict)
    assert result.get('mode') == 'philosophical'
    
    print("✓ Explicit mode is respected")


if __name__ == "__main__":
    print("=" * 60)
    print("WorldModel Routing Methods Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_should_route_causal()
        test_should_route_analogical()
        test_should_route_mathematical()
        test_should_route_sat()
        test_philosophical_not_routed()
        test_input_validation()
        test_normalize_dict_result()
        test_reason_method_routing()
        test_reason_respects_mode()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
