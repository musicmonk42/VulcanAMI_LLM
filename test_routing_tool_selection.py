#!/usr/bin/env python3
"""
Test script for MEDIUM Priority Fix #5: _select_reasoning_tools method
Tests proper engine selection and prevents misrouting of technical queries.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vulcan.routing.query_router import QueryAnalyzer, ProcessingPlan, QueryType


def test_sat_query():
    """Test SAT queries route to symbolic engine"""
    analyzer = QueryAnalyzer()
    
    # Create a mock plan for SAT query
    plan = ProcessingPlan(
        query_id="test_sat_1",
        original_query="Is the formula (P → Q) ∧ (Q → R) ∧ ¬R satisfiable?",
        source="user",
        learning_mode=None,  # Will be set by route_query
        query_type=QueryType.REASONING,
        detected_patterns=['sat_problem', 'logic_symbols'],
        complexity_score=0.7,
        uncertainty_score=0.3
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ SAT Query Test: {tools}")
    assert tools == ['symbolic'], f"Expected ['symbolic'], got {tools}"
    print("  PASS: SAT query correctly routed to symbolic engine\n")


def test_causal_query():
    """Test causal queries route to causal engine"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_causal_1",
        original_query="What is the causal effect of confounding vs causation?",
        source="user",
        learning_mode=None,
        query_type=QueryType.REASONING,
        detected_patterns=['causal_reasoning'],
        complexity_score=0.6,
        uncertainty_score=0.4
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Causal Query Test: {tools}")
    assert tools == ['causal'], f"Expected ['causal'], got {tools}"
    print("  PASS: Causal query correctly routed to causal engine\n")


def test_analogical_query():
    """Test analogical queries route to analogical engine"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_analog_1",
        original_query="Map the deep structure from domain S to domain T using structure mapping",
        source="user",
        learning_mode=None,
        query_type=QueryType.REASONING,
        detected_patterns=['analogical_reasoning'],
        complexity_score=0.5,
        uncertainty_score=0.3
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Analogical Query Test: {tools}")
    assert tools == ['analogical'], f"Expected ['analogical'], got {tools}"
    print("  PASS: Analogical query correctly routed to analogical engine\n")


def test_mathematical_query():
    """Test mathematical queries route to mathematical engine"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_math_1",
        original_query="Calculate the sum ∑(2k-1) from k=1 to n",
        source="user",
        learning_mode=None,
        query_type=QueryType.EXECUTION,
        detected_patterns=['math_calculation'],
        complexity_score=0.4,
        uncertainty_score=0.2
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Mathematical Query Test: {tools}")
    assert tools == ['mathematical'], f"Expected ['mathematical'], got {tools}"
    print("  PASS: Mathematical query correctly routed to mathematical engine\n")


def test_complex_math_query():
    """Test complex math queries route to both mathematical and symbolic engines"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_math_2",
        original_query="Prove the theorem: for all n, the sum of first n odd numbers equals n²",
        source="user",
        learning_mode=None,
        query_type=QueryType.REASONING,
        detected_patterns=['math_proof'],
        complexity_score=0.8,
        uncertainty_score=0.3
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Complex Math Query Test: {tools}")
    assert tools == ['mathematical', 'symbolic'], f"Expected ['mathematical', 'symbolic'], got {tools}"
    print("  PASS: Complex math query correctly routed to mathematical + symbolic engines\n")


def test_philosophical_query():
    """Test philosophical queries route to philosophical and world_model engines"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_phil_1",
        original_query="What is the trolley problem and what are the ethical implications?",
        source="user",
        learning_mode=None,
        query_type=QueryType.PHILOSOPHICAL,
        detected_patterns=['philosophical_query'],
        complexity_score=0.5,
        uncertainty_score=0.4
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Philosophical Query Test: {tools}")
    assert tools == ['philosophical', 'world_model'], f"Expected ['philosophical', 'world_model'], got {tools}"
    print("  PASS: Philosophical query correctly routed to philosophical + world_model engines\n")


def test_ensemble_high_complexity():
    """Test very complex queries get ensemble of tools"""
    analyzer = QueryAnalyzer()
    
    plan = ProcessingPlan(
        query_id="test_ensemble_1",
        original_query="Analyze the complex multi-factor system with high uncertainty",
        source="user",
        learning_mode=None,
        query_type=QueryType.REASONING,
        detected_patterns=['complex_system'],
        complexity_score=0.85,
        uncertainty_score=0.6
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Ensemble Query Test: {tools}")
    assert tools == ['causal', 'probabilistic', 'world_model'], \
        f"Expected ['causal', 'probabilistic', 'world_model'], got {tools}"
    print("  PASS: High complexity + uncertainty correctly triggers ensemble\n")


def test_priority_ordering():
    """Test that SAT takes priority over other patterns"""
    analyzer = QueryAnalyzer()
    
    # Plan with both SAT and mathematical patterns - SAT should win
    plan = ProcessingPlan(
        query_id="test_priority_1",
        original_query="Is (P → Q) ∧ R satisfiable? Calculate the number of models.",
        source="user",
        learning_mode=None,
        query_type=QueryType.REASONING,
        detected_patterns=['sat_problem', 'math_calculation'],
        complexity_score=0.7,
        uncertainty_score=0.3
    )
    
    tools = analyzer._select_reasoning_tools(plan)
    print(f"✓ Priority Ordering Test: {tools}")
    assert tools == ['symbolic'], f"Expected ['symbolic'] (SAT priority), got {tools}"
    print("  PASS: SAT correctly takes priority over mathematical patterns\n")


def main():
    """Run all tests"""
    print("="*70)
    print("Testing MEDIUM Priority Fix #5: _select_reasoning_tools")
    print("="*70 + "\n")
    
    try:
        test_sat_query()
        test_causal_query()
        test_analogical_query()
        test_mathematical_query()
        test_complex_math_query()
        test_philosophical_query()
        test_ensemble_high_complexity()
        test_priority_ordering()
        
        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
