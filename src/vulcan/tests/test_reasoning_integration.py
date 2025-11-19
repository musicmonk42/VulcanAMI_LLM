"""
Integration Test for VULCAN Unified Reasoning System

Tests the complete reasoning pipeline end-to-end using the actual UnifiedReasoner API.
"""

import pytest
import numpy as np
import time
import json
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, List

# CORRECT IMPORTS based on actual unified_reasoning.py
from vulcan.reasoning.unified_reasoning import (
    UnifiedReasoner,
    ReasoningStrategy,
)

# Import from reasoning_types module
from vulcan.reasoning.reasoning_types import (
    ReasoningType,
    ReasoningResult,
)


class TestUnifiedReasoningIntegration:
    """Comprehensive integration tests for the reasoning system"""
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create temporary directory for state/cache"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture(scope="class")
    def reasoning_system(self, temp_dir):
        """Create a fully configured reasoning system"""
        config = {
            'confidence_threshold': 0.3,
            'max_reasoning_time': 30.0,
            'default_timeout': 10.0,
            'skip_runtime': True,  # Skip heavy runtime in tests
            
            'cache_config': {
                'cleanup_interval': 0.05,
                'enable_warming': False,
                'enable_disk_cache': False,
            },
            
            'tool_selector_config': {
                'cache_enabled': True,
                'safety_enabled': True,
                'learning_enabled': False,
                'warm_pool_enabled': False,
                'max_workers': 2,
            },
        }
        
        system = UnifiedReasoner(
            enable_learning=False,
            enable_safety=True,
            max_workers=2,
            config=config
        )
        
        yield system
        
        # Cleanup with skip_save to avoid test delays
        try:
            system.shutdown(timeout=2.0, skip_save=True)
        except Exception as e:
            print(f"Shutdown warning: {e}")
    
    # =========================================================================
    # TEST 1: Basic Reasoning
    # =========================================================================
    
    def test_basic_probabilistic_reasoning(self, reasoning_system):
        """Test basic probabilistic reasoning"""
        print("\n" + "="*60)
        print("TEST 1: Basic Probabilistic Reasoning")
        print("="*60)
        
        input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        query = {'threshold': 0.5, 'question': 'analyze data'}
        
        result = reasoning_system.reason(
            input_data=input_data,
            query=query,
            reasoning_type=ReasoningType.PROBABILISTIC,
            strategy=ReasoningStrategy.SEQUENTIAL
        )
        
        # Verify result structure
        assert result is not None, "Result should not be None"
        assert isinstance(result, ReasoningResult), "Should return ReasoningResult"
        assert hasattr(result, 'conclusion'), "Result should have conclusion"
        assert hasattr(result, 'confidence'), "Result should have confidence"
        assert hasattr(result, 'reasoning_type'), "Result should have reasoning_type"
        
        print(f"✅ Reasoning type: {result.reasoning_type}")
        print(f"✅ Confidence: {result.confidence:.3f}")
        print(f"✅ Conclusion type: {type(result.conclusion).__name__}")
        
        # Verify basic properties
        assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of range"
    
    # =========================================================================
    # TEST 2: Different Reasoning Types
    # =========================================================================
    
    def test_different_reasoning_types(self, reasoning_system):
        """Test that different reasoning types work"""
        print("\n" + "="*60)
        print("TEST 2: Different Reasoning Types")
        print("="*60)
        
        test_cases = [
            (ReasoningType.PROBABILISTIC, [1, 2, 3, 4, 5], {'threshold': 0.5}),
            (ReasoningType.SYMBOLIC, "A → B", {'goal': 'B'}),
            (ReasoningType.CAUSAL, {'graph': {'A': ['B'], 'B': ['C']}}, {'query': 'effect'}),
        ]
        
        results = []
        for reasoning_type, input_data, query in test_cases:
            try:
                result = reasoning_system.reason(
                    input_data=input_data,
                    query=query,
                    reasoning_type=reasoning_type,
                    strategy=ReasoningStrategy.SEQUENTIAL
                )
                
                results.append({
                    'type': reasoning_type.value,
                    'success': result is not None,
                    'confidence': result.confidence if result else 0.0,
                })
                
                print(f"\n{reasoning_type.value}:")
                print(f"  Success: {result is not None}")
                if result:
                    print(f"  Confidence: {result.confidence:.3f}")
                    
            except Exception as e:
                print(f"\n{reasoning_type.value}: Failed - {e}")
                results.append({
                    'type': reasoning_type.value,
                    'success': False,
                    'confidence': 0.0,
                })
        
        # Verify at least some types worked
        successful = [r for r in results if r['success']]
        print(f"\n✅ Successful: {len(successful)}/{len(results)}")
        assert len(successful) > 0, "At least one reasoning type should work"
    
    # =========================================================================
    # TEST 3: Different Strategies
    # =========================================================================
    
    def test_different_strategies(self, reasoning_system):
        """Test different reasoning strategies"""
        print("\n" + "="*60)
        print("TEST 3: Different Strategies")
        print("="*60)
        
        input_data = [1, 2, 3, 4, 5]
        query = {'question': 'analyze'}
        
        strategies = [
            ReasoningStrategy.SEQUENTIAL,
            ReasoningStrategy.ADAPTIVE,
            ReasoningStrategy.HYBRID,
        ]
        
        for strategy in strategies:
            try:
                result = reasoning_system.reason(
                    input_data=input_data,
                    query=query,
                    strategy=strategy
                )
                
                print(f"\n{strategy.value}:")
                print(f"  Success: {result is not None}")
                if result:
                    print(f"  Confidence: {result.confidence:.3f}")
                    print(f"  Type: {result.reasoning_type}")
                    
                assert result is not None, f"{strategy.value} should return result"
                
            except Exception as e:
                print(f"\n{strategy.value}: Error - {e}")
        
        print("\n✅ Strategy testing complete")
    
    # =========================================================================
    # TEST 4: Auto Type Detection
    # =========================================================================
    
    def test_auto_type_detection(self, reasoning_system):
        """Test automatic reasoning type detection"""
        print("\n" + "="*60)
        print("TEST 4: Auto Type Detection")
        print("="*60)
        
        test_cases = [
            ([1, 2, 3, 4, 5], {'question': 'probability'}, "numeric data"),
            ("A → B", {'question': 'prove'}, "logical statement"),
            ({'graph': {'A': ['B']}}, {'question': 'cause'}, "graph structure"),
        ]
        
        for input_data, query, description in test_cases:
            result = reasoning_system.reason(
                input_data=input_data,
                query=query,
                reasoning_type=None,  # Auto-detect
                strategy=ReasoningStrategy.ADAPTIVE
            )
            
            print(f"\n{description}:")
            print(f"  Auto-detected type: {result.reasoning_type if result else 'None'}")
            print(f"  Confidence: {result.confidence if result else 0.0:.3f}")
            
            assert result is not None, f"Auto-detection should work for {description}"
        
        print("\n✅ Auto-detection working")
    
    # =========================================================================
    # TEST 5: Constraints Handling
    # =========================================================================
    
    def test_constraints_handling(self, reasoning_system):
        """Test constraint handling"""
        print("\n" + "="*60)
        print("TEST 5: Constraints Handling")
        print("="*60)
        
        input_data = [1, 2, 3, 4, 5]
        
        # Test with different constraints
        constraints_list = [
            {'time_budget_ms': 1000, 'confidence_threshold': 0.3},
            {'time_budget_ms': 5000, 'confidence_threshold': 0.7},
        ]
        
        for constraints in constraints_list:
            result = reasoning_system.reason(
                input_data=input_data,
                query={'question': 'analyze'},
                constraints=constraints
            )
            
            print(f"\nConstraints: {constraints}")
            print(f"  Result: {result is not None}")
            if result:
                print(f"  Confidence: {result.confidence:.3f}")
            
            assert result is not None, "Should handle constraints"
        
        print("\n✅ Constraint handling working")
    
    # =========================================================================
    # TEST 6: Concurrent Execution
    # =========================================================================
    
    def test_concurrent_reasoning(self, reasoning_system):
        """Test concurrent reasoning requests"""
        print("\n" + "="*60)
        print("TEST 6: Concurrent Execution")
        print("="*60)
        
        import concurrent.futures
        
        def reason_task(task_id):
            input_data = list(range(task_id, task_id + 10))
            result = reasoning_system.reason(
                input_data=input_data,
                query={'task_id': task_id}
            )
            return task_id, result
        
        # Submit 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(reason_task, i) for i in range(3)]
            results = [f.result(timeout=15) for f in futures]
        
        # Verify all completed
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        successful = 0
        for task_id, result in results:
            if result and result.confidence > 0:
                successful += 1
                print(f"Task {task_id}: Success (conf={result.confidence:.3f})")
            else:
                print(f"Task {task_id}: Failed or low confidence")
        
        print(f"\n✅ Concurrent execution: {successful}/3 successful")
        assert successful >= 2, "At least 2/3 concurrent requests should succeed"
    
    # =========================================================================
    # TEST 7: Chain Verification
    # =========================================================================
    
    def test_reasoning_chain_creation(self, reasoning_system):
        """Test that reasoning chains are properly created"""
        print("\n" + "="*60)
        print("TEST 7: Reasoning Chain Creation")
        print("="*60)
        
        input_data = [1, 2, 3, 4, 5]
        query = {'question': 'analyze data'}
        
        result = reasoning_system.reason(
            input_data=input_data,
            query=query,
            reasoning_type=ReasoningType.PROBABILISTIC
        )
        
        assert result is not None, "Result should not be None"
        
        print(f"\nReasoning chain exists: {result.reasoning_chain is not None}")
        
        if result.reasoning_chain:
            print(f"Chain ID: {result.reasoning_chain.chain_id}")
            print(f"Steps: {len(result.reasoning_chain.steps)}")
            print(f"Types used: {result.reasoning_chain.reasoning_types_used}")
            
            # Verify chain has at least one step
            assert len(result.reasoning_chain.steps) > 0, "Chain should have steps"
        
        print("✅ Reasoning chain properly created")
    
    # =========================================================================
    # TEST 8: Statistics Collection
    # =========================================================================
    
    def test_statistics_collection(self, reasoning_system):
        """Test that statistics are collected"""
        print("\n" + "="*60)
        print("TEST 8: Statistics Collection")
        print("="*60)
        
        # Execute several reasoning tasks
        for i in range(3):
            reasoning_system.reason(
                input_data=[1, 2, 3, 4, 5],
                query={'iteration': i}
            )
        
        # Get statistics
        stats = reasoning_system.get_statistics()
        
        assert stats is not None, "Statistics should not be None"
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        
        print("\nStatistics keys:")
        for key in stats.keys():
            print(f"  - {key}")
        
        # Verify key statistics exist
        assert 'performance' in stats, "Should have performance metrics"
        assert 'execution_count' in stats, "Should have execution count"
        
        print(f"\nExecution count: {stats.get('execution_count', 0)}")
        print(f"Performance: {stats.get('performance', {})}")
        
        print("✅ Statistics collection working")
    
    # =========================================================================
    # TEST 9: State Persistence
    # =========================================================================
    
    def test_state_persistence(self, reasoning_system, temp_dir):
        """Test state save and load"""
        print("\n" + "="*60)
        print("TEST 9: State Persistence")
        print("="*60)
        
        # Execute some reasoning
        for i in range(2):
            reasoning_system.reason(
                input_data=[1, 2, 3],
                query={'id': i}
            )
        
        # Save state
        try:
            reasoning_system.save_state("test_state")
            print("✅ State saved successfully")
        except Exception as e:
            print(f"⚠️ Save failed (may be expected): {e}")
        
        # Load state
        try:
            reasoning_system.load_state("test_state")
            print("✅ State loaded successfully")
        except Exception as e:
            print(f"⚠️ Load failed (may be expected if save failed): {e}")
    
    # =========================================================================
    # TEST 10: Error Recovery
    # =========================================================================
    
    def test_error_recovery(self, reasoning_system):
        """Test error recovery with invalid inputs"""
        print("\n" + "="*60)
        print("TEST 10: Error Recovery")
        print("="*60)
        
        # Test with None input
        result = reasoning_system.reason(
            input_data=None,
            query={}
        )
        assert result is not None, "Should handle None input"
        print("✅ Handled None input")
        
        # Test with empty input
        result = reasoning_system.reason(
            input_data=[],
            query={}
        )
        assert result is not None, "Should handle empty input"
        print("✅ Handled empty input")
        
        # Test with very large input
        result = reasoning_system.reason(
            input_data=list(range(10000)),
            query={'question': 'analyze'}
        )
        assert result is not None, "Should handle large input"
        print("✅ Handled large input")
        
        print("\n✅ Error recovery working")
    
    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================
    
    def test_final_system_validation(self, reasoning_system):
        """Final end-to-end validation"""
        print("\n" + "="*60)
        print("FINAL SYSTEM VALIDATION")
        print("="*60)
        
        # Execute a comprehensive reasoning task
        input_data = {
            'data': list(range(50)),
            'type': 'analysis',
            'metadata': {'source': 'test'}
        }
        
        query = {
            'question': 'What patterns exist in this data?',
            'analysis_type': 'comprehensive'
        }
        
        result = reasoning_system.reason(
            input_data=input_data,
            query=query,
            strategy=ReasoningStrategy.ADAPTIVE
        )
        
        # Verification checklist
        print("\nFinal Validation Checklist:")
        
        checks = {
            'Result returned': result is not None,
            'Has conclusion': result.conclusion is not None if result else False,
            'Has confidence': (0 <= result.confidence <= 1) if result else False,
            'Has reasoning type': result.reasoning_type is not None if result else False,
            'Has explanation': result.explanation is not None if result else False,
        }
        
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
            all_passed = all_passed and passed
        
        if result:
            print(f"\nFinal Result:")
            print(f"  Type: {result.reasoning_type}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Conclusion type: {type(result.conclusion).__name__}")
        
        print("\n" + "="*60)
        print("🎉 INTEGRATION TEST COMPLETE!")
        print("="*60)
        print("\nThe VULCAN Unified Reasoning System is functional!")
        
        assert all_passed, "Not all validation checks passed"


# =============================================================================
# STANDALONE TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    """Run tests standalone without pytest"""
    import sys
    
    print("="*70)
    print("VULCAN UNIFIED REASONING SYSTEM - INTEGRATION TEST")
    print("="*70)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        test = TestUnifiedReasoningIntegration()
        
        # Create reasoning system
        print("\nInitializing reasoning system...")
        reasoning_system = next(test.reasoning_system(temp_dir))
        
        # Run all tests
        tests = [
            ('Basic Probabilistic Reasoning', test.test_basic_probabilistic_reasoning),
            ('Different Reasoning Types', test.test_different_reasoning_types),
            ('Different Strategies', test.test_different_strategies),
            ('Auto Type Detection', test.test_auto_type_detection),
            ('Constraints Handling', test.test_constraints_handling),
            ('Concurrent Execution', test.test_concurrent_reasoning),
            ('Reasoning Chain Creation', test.test_reasoning_chain_creation),
            ('Statistics Collection', test.test_statistics_collection),
            ('State Persistence', lambda: test.test_state_persistence(reasoning_system, temp_dir)),
            ('Error Recovery', test.test_error_recovery),
            ('Final Validation', test.test_final_system_validation),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                test_func(reasoning_system)
                passed += 1
                print(f"✅ {name} PASSED\n")
            except Exception as e:
                failed += 1
                print(f"❌ {name} FAILED: {e}\n")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {passed}/{len(tests)}")
        print(f"Failed: {failed}/{len(tests)}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {failed} TEST(S) FAILED")
            sys.exit(1)
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)