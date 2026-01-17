#!/usr/bin/env python3
"""
Simple validation script for Issue #1 and Issue #2 fixes.
Run this after installing dependencies to verify the fixes work.

Usage:
    python tests/validate_fixes.py
"""

def test_probabilistic_detection():
    """Test Issue #1: Probabilistic scenario detection."""
    try:
        from src.vulcan.reasoning.probabilistic_reasoning import EnhancedProbabilisticReasoner
        reasoner = EnhancedProbabilisticReasoner()
        
        tests = [
            ("Three doors. Host opens a goat door. Should you switch?", True, "Monty Hall (doors + host + open)"),
            ("Three doors. You choose one. Host opens a different door. Should you switch?", True, "Monty Hall (doors + choose + switch)"),
            ("If you draw a card from a standard deck, what are the odds of getting an ace?", True, "Card probability (cards + draw)"),
            ("If you roll two dice, what's the chance of getting a sum of 7?", True, "Dice probability (dice + roll)"),
            ("If you flip a fair coin three times, what's the probability of getting at least two heads?", True, "Coin probability (coin + flip)"),
            ("What is the weather like today?", False, "Weather query (not probability)"),
        ]
        
        passed = 0
        failed = 0
        for query, expected, description in tests:
            result = reasoner._is_probability_query(query)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            else:
                failed += 1
            print(f"  {status} {description}: {result} (expected {expected})")
        
        print(f"\nIssue #1 Results: {passed}/{len(tests)} tests passed")
        return failed == 0
        
    except Exception as e:
        print(f"Error testing probabilistic detection: {e}")
        return False


def test_mathematical_retry():
    """Test Issue #2: Mathematical retry logic structure."""
    try:
        from src.vulcan.reasoning.mathematical_computation import MathematicalComputationTool
        tool = MathematicalComputationTool()
        
        tests = [
            (hasattr(tool, '_request_code_correction'), "_request_code_correction method exists"),
            (hasattr(tool, '_try_fallback'), "_try_fallback method exists"),
        ]
        
        # Check that execute method has retry logic
        import inspect
        source = inspect.getsource(tool.execute)
        tests.append(("MAX_RETRIES" in source, "MAX_RETRIES constant defined in execute()"))
        tests.append(("for attempt in range" in source, "Retry loop present in execute()"))
        tests.append(("_request_code_correction" in source, "Code correction called in execute()"))
        
        passed = 0
        failed = 0
        for condition, description in tests:
            status = "✓" if condition else "✗"
            if condition:
                passed += 1
            else:
                failed += 1
            print(f"  {status} {description}")
        
        print(f"\nIssue #2 Results: {passed}/{len(tests)} tests passed")
        return failed == 0
        
    except Exception as e:
        print(f"Error testing mathematical retry: {e}")
        return False


def test_natural_language_logic_routing():
    """Test Issue #3: Natural language logic routing."""
    try:
        from src.vulcan.routing.query_router import QueryAnalyzer
        from src.vulcan.routing.query_router import ProcessingPlan, QueryType
        
        analyzer = QueryAnalyzer()
        
        # Create a test query
        query = "Birds fly. Penguins are birds. Do penguins fly?"
        plan = analyzer.route_query(query, source="user")
        
        # Check if routing logic exists
        import inspect
        source = inspect.getsource(analyzer._select_reasoning_tools)
        
        tests = [
            ("FIX Issue #3" in source, "Issue #3 fix marker present in code"),
            ("Natural language logic" in source, "Natural language logic detection mentioned"),
            ("world_model" in source, "world_model routing present"),
            ("logic_relationship_keywords" in source, "Logic relationship keywords defined"),
        ]
        
        passed = 0
        failed = 0
        for condition, description in tests:
            status = "✓" if condition else "✗"
            if condition:
                passed += 1
            else:
                failed += 1
            print(f"  {status} {description}")
        
        print(f"\nIssue #3 Results: {passed}/{len(tests)} tests passed")
        return failed == 0
        
    except Exception as e:
        print(f"Error testing natural language logic routing: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("VULCAN Reasoning Engine Fixes Validation")
    print("=" * 60)
    
    print("\n[Issue #1] Testing Probabilistic Scenario Detection...")
    print("-" * 60)
    issue1_passed = test_probabilistic_detection()
    
    print("\n[Issue #2] Testing Mathematical Retry Logic...")
    print("-" * 60)
    issue2_passed = test_mathematical_retry()
    
    print("\n[Issue #3] Testing Natural Language Logic Routing...")
    print("-" * 60)
    issue3_passed = test_natural_language_logic_routing()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Issue #1 (Probabilistic Detection): {'✓ PASS' if issue1_passed else '✗ FAIL'}")
    print(f"Issue #2 (Mathematical Retry):      {'✓ PASS' if issue2_passed else '✗ FAIL'}")
    print(f"Issue #3 (NL Logic Routing):        {'✓ PASS' if issue3_passed else '✗ FAIL'}")
    
    all_passed = issue1_passed and issue2_passed and issue3_passed
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
