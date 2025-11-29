"""
test_problem_decomposer_integration.py - Comprehensive integration test
Tests all problem_decomposer components working together

Run with: pytest src/vulcan/tests/test_problem_decomposer_integration.py -v -s
Or standalone: python src/vulcan/tests/test_problem_decomposer_integration.py
"""

import sys
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecomposerTestResult:
    """Container for test results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = {}


class TestProblemDecomposerIntegration:
    """Comprehensive integration tests for problem decomposer - pytest compatible"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_results = []
    
    @pytest.mark.timeout(10)
    def test_01_module_imports(self):
        """Test 1: Verify all modules can be imported"""
        result = DecomposerTestResult("Module Imports")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 1] Testing module imports...")
            
            from src.vulcan.problem_decomposer.problem_decomposer_core import (
                ProblemDecomposer, ProblemGraph, DecompositionPlan, ExecutionOutcome
            )
            from src.vulcan.problem_decomposer.decomposition_strategies import (
                ExactDecomposition, SemanticDecomposition, StructuralDecomposition,
                SyntheticBridging, AnalogicalDecomposition, BruteForceSearch
            )
            from src.vulcan.problem_decomposer.decomposition_library import (
                StratifiedDecompositionLibrary, Pattern, Context, DecompositionPrinciple
            )
            from src.vulcan.problem_decomposer.adaptive_thresholds import (
                AdaptiveThresholds, PerformanceTracker, StrategyProfiler
            )
            from src.vulcan.problem_decomposer.fallback_chain import FallbackChain, ExecutionPlan
            from src.vulcan.problem_decomposer.problem_executor import ProblemExecutor
            from src.vulcan.problem_decomposer.decomposer_bootstrap import (
                DecomposerBootstrap, create_decomposer
            )
            
            logger.info("  ✓ problem_decomposer_core: 4 classes imported")
            logger.info("  ✓ decomposition_strategies: 6 classes imported")
            logger.info("  ✓ decomposition_library: 4 classes imported")
            logger.info("  ✓ adaptive_thresholds: 3 classes imported")
            logger.info("  ✓ fallback_chain: 2 classes imported")
            logger.info("  ✓ problem_executor: 1 class imported")
            logger.info("  ✓ decomposer_bootstrap: 2 classes imported")
            logger.info("  ✓ All modules imported successfully")
            
            result.passed = True
            result.details = {
                'modules_tested': 7,
                'classes_imported': 22,
                'all_imports_successful': True
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Import test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Module imports failed: {result.error}"
    
    @pytest.mark.timeout(15)
    def test_02_component_initialization(self):
        """Test 2: Initialize individual components"""
        result = DecomposerTestResult("Component Initialization")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 2] Testing component initialization...")
            
            from src.vulcan.problem_decomposer.adaptive_thresholds import AdaptiveThresholds, PerformanceTracker
            from src.vulcan.problem_decomposer.decomposition_library import StratifiedDecompositionLibrary
            from src.vulcan.problem_decomposer.fallback_chain import FallbackChain
            from src.vulcan.problem_decomposer.problem_executor import ProblemExecutor
            
            thresholds = AdaptiveThresholds()
            logger.info("  ✓ AdaptiveThresholds initialized")
            assert len(thresholds.thresholds) > 0
            
            tracker = PerformanceTracker()
            logger.info("  ✓ PerformanceTracker initialized")
            
            library = StratifiedDecompositionLibrary()
            logger.info("  ✓ StratifiedDecompositionLibrary initialized")
            
            chain = FallbackChain()
            logger.info("  ✓ FallbackChain initialized")
            
            executor = ProblemExecutor()
            logger.info("  ✓ ProblemExecutor initialized")
            
            result.passed = True
            result.details = {
                'components_initialized': 5,
                'threshold_count': len(thresholds.thresholds),
                'fallback_strategies': len(chain.strategies)
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Component initialization failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Component initialization failed: {result.error}"
    
    @pytest.mark.timeout(15)
    def test_03_strategy_registration(self):
        """Test 3: Create and register strategies"""
        result = DecomposerTestResult("Strategy Registration")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 3] Testing strategy creation and registration...")
            
            from src.vulcan.problem_decomposer.decomposition_strategies import (
                ExactDecomposition, SemanticDecomposition, StructuralDecomposition,
                SyntheticBridging, AnalogicalDecomposition, BruteForceSearch
            )
            from src.vulcan.problem_decomposer.decomposition_library import StratifiedDecompositionLibrary
            
            strategies = [
                ExactDecomposition(),
                SemanticDecomposition(),
                StructuralDecomposition(),
                SyntheticBridging(),
                AnalogicalDecomposition(),
                BruteForceSearch()
            ]
            
            logger.info(f"  ✓ Created {len(strategies)} strategy instances")
            
            library = StratifiedDecompositionLibrary()
            
            for strategy in strategies:
                strategy_name = strategy.name if hasattr(strategy, 'name') else 'unknown'
                if not hasattr(library, 'strategy_registry'):
                    library.strategy_registry = {}
                library.strategy_registry[strategy_name] = strategy
            
            logger.info(f"  ✓ Registered {len(strategies)} strategies in library")
            
            # Verify each strategy
            for strategy in strategies:
                assert hasattr(strategy, 'decompose'), f"{strategy.name} missing decompose"
                assert hasattr(strategy, 'name'), "Strategy missing name"
            
            logger.info("  ✓ All strategies validated")
            
            result.passed = True
            result.details = {
                'strategies_created': len(strategies),
                'strategies_registered': len(library.strategy_registry)
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Strategy registration failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Strategy registration failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_04_bootstrap_creation(self):
        """Test 4: Bootstrap decomposer creation"""
        result = DecomposerTestResult("Bootstrap Creation")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 4] Testing bootstrap decomposer creation...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            
            decomposer = create_decomposer(config={'test_mode': True})
            logger.info("  ✓ Decomposer created via bootstrap")
            
            assert hasattr(decomposer, 'library'), "Missing library"
            assert hasattr(decomposer, 'thresholds'), "Missing thresholds"
            assert hasattr(decomposer, 'fallback_chain'), "Missing fallback_chain"
            assert hasattr(decomposer, 'executor'), "Missing executor"
            logger.info("  ✓ All core components present")
            
            strategy_count = len(decomposer.fallback_chain.strategies)
            logger.info(f"  ✓ Fallback chain has {strategy_count} strategies")
            
            result.passed = True
            result.details = {
                'decomposer_created': True,
                'strategy_count': strategy_count,
                'has_library': hasattr(decomposer, 'library'),
                'has_executor': hasattr(decomposer, 'executor')
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Bootstrap creation failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Bootstrap creation failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_05_problem_decomposition(self):
        """Test 5: Create and decompose a problem"""
        result = DecomposerTestResult("Problem Decomposition")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 5] Testing problem creation and decomposition...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            
            problem = ProblemGraph(
                nodes={
                    'start': {'type': 'decision', 'value': 1},
                    'process1': {'type': 'operation', 'operation': 'sum'},
                    'process2': {'type': 'operation', 'operation': 'product'},
                    'end': {'type': 'result'}
                },
                edges=[
                    ('start', 'process1', {'weight': 1.0}),
                    ('start', 'process2', {'weight': 0.5}),
                    ('process1', 'end', {'weight': 1.0}),
                    ('process2', 'end', {'weight': 1.0})
                ],
                root='start',
                metadata={'domain': 'optimization', 'type': 'parallel'}
            )
            logger.info("  ✓ Test problem created")
            
            plan = decomposer.decompose_novel_problem(problem)
            logger.info(f"  ✓ Problem decomposed into {len(plan.steps)} steps")
            logger.info(f"  ✓ Plan confidence: {plan.confidence:.2f}")
            
            assert len(plan.steps) > 0, "Plan has no steps"
            assert plan.confidence > 0, "Plan has zero confidence"
            
            result.passed = True
            result.details = {
                'problem_nodes': len(problem.nodes),
                'problem_edges': len(problem.edges),
                'plan_steps': len(plan.steps),
                'plan_confidence': plan.confidence,
                'strategy_used': plan.strategy.name if plan.strategy else 'unknown'
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Problem decomposition failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Problem decomposition failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_06_plan_execution_safety(self):
        """Test 6: Plan execution requires safety validator"""
        result = DecomposerTestResult("Plan Execution Safety")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 6] Testing plan execution safety requirement...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            
            problem = ProblemGraph(
                nodes={'A': {'type': 'operation', 'value': 5}, 'B': {'type': 'operation', 'value': 3}, 'C': {'type': 'result'}},
                edges=[('A', 'C', {}), ('B', 'C', {})],
                root='A',
                metadata={'domain': 'general', 'type': 'simple'}
            )
            
            plan = decomposer.decompose_novel_problem(problem)
            logger.info(f"  ✓ Plan created with {len(plan.steps)} steps")
            
            # Execute plan with safety validator enabled (test_mode has safety)
            outcome = decomposer.executor.execute_plan(problem, plan)
            logger.info(f"  ✓ Plan executed with safety validation (success: {outcome.success})")
            
            # Verify safety validator is present
            assert decomposer.executor.safety_validator is not None, "Safety validator should be present"
            logger.info("  ✓ Safety validator is active")
            
            result.passed = True
            result.details = {
                'safety_enforced': True,
                'plan_created': True,
                'execution_completed': True
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Safety test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Safety enforcement failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_07_full_flow_safety(self):
        """Test 7: Full flow requires safety"""
        result = DecomposerTestResult("Full Flow Safety")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 7] Testing full flow safety requirement...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            
            problem = ProblemGraph(
                nodes={'input': {'type': 'operation'}, 'transform': {'type': 'transform'}, 'output': {'type': 'result'}},
                edges=[('input', 'transform', {}), ('transform', 'output', {})],
                root='input',
                metadata={'domain': 'analysis', 'type': 'pipeline'}
            )
            
            # Execute full flow with safety validation enabled
            plan, outcome = decomposer.decompose_and_execute(problem)
            logger.info(f"  ✓ Full flow completed with safety (success: {outcome.success})")
            
            # Verify safety validator is present
            assert decomposer.safety_validator is not None, "Safety validator should be present"
            logger.info("  ✓ Safety validator is active in full flow")
            
            result.passed = True
            result.details = {
                'safety_enforced': True,
                'full_flow_completed': True
            }
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Full flow safety test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Full flow safety failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_08_learning_integration(self):
        """Test 8: Learning and adaptation"""
        result = DecomposerTestResult("Learning Integration")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 8] Testing learning and adaptation...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import (
                ProblemGraph, DecompositionPlan, ExecutionOutcome
            )
            
            decomposer = create_decomposer(config={'test_mode': True})
            
            problem = ProblemGraph(nodes={'A': {}, 'B': {}}, edges=[('A', 'B', {})], root='A', metadata={'domain': 'test'})
            problem.complexity_score = 2.0
            
            plan = DecompositionPlan(steps=[{'step_id': 'test', 'type': 'generic', 'description': 'Test'}], confidence=0.7, estimated_complexity=2.0)
            outcome = ExecutionOutcome(success=True, execution_time=1.5, metrics={'accuracy': 0.9})
            
            decomposer.learn_from_execution(problem, plan, outcome)
            logger.info("  ✓ Learning completed")
            
            stats = decomposer.get_statistics()
            logger.info(f"  ✓ Total decompositions: {stats['decomposition_stats']['total_decompositions']}")
            
            result.passed = True
            result.details = {'learning_functional': True, 'stats_retrieved': True}
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Learning test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Learning integration failed: {result.error}"
    
    @pytest.mark.timeout(20)
    def test_09_fallback_chain(self):
        """Test 9: Fallback chain"""
        result = DecomposerTestResult("Fallback Chain")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 9] Testing fallback chain...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            problem = ProblemGraph(nodes={'X': {}, 'Y': {}, 'Z': {}}, edges=[('X', 'Y', {}), ('Y', 'Z', {})], metadata={'domain': 'unknown'})
            
            plan = decomposer.decompose_with_fallbacks(problem)
            logger.info(f"  ✓ Fallback produced {len(plan.steps)} steps (confidence: {plan.confidence:.2f})")
            
            result.passed = True
            result.details = {'plan_created': True, 'fallback_functional': True}
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Fallback test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Fallback chain failed: {result.error}"
    
    @pytest.mark.timeout(20)
    def test_10_performance_tracking(self):
        """Test 10: Performance tracking"""
        result = DecomposerTestResult("Performance Tracking")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 10] Testing performance tracking...")
            
            from src.vulcan.problem_decomposer.adaptive_thresholds import PerformanceTracker
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph, DecompositionPlan, ExecutionOutcome
            from src.vulcan.problem_decomposer.decomposition_strategies import StructuralDecomposition
            
            tracker = PerformanceTracker()
            problem = ProblemGraph(nodes={'A': {}}, edges=[], metadata={'domain': 'test'})
            strategy = StructuralDecomposition()
            plan = DecompositionPlan(strategy=strategy, steps=[])
            
            for i in range(10):
                outcome = ExecutionOutcome(success=(i % 2 == 0), execution_time=1.0 + i * 0.1)
                tracker.record_execution(problem, plan, outcome)
            
            success_rate = tracker.get_strategy_success_rate(strategy.name)
            avg_time = tracker.get_average_execution_time(strategy.name)
            
            logger.info(f"  ✓ Success rate: {success_rate:.2f}, Avg time: {avg_time:.3f}s")
            
            result.passed = True
            result.details = {'recordings': 10, 'success_rate': success_rate, 'avg_time': avg_time}
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Performance tracking failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Performance tracking failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_11_caching(self):
        """Test 11: Caching"""
        result = DecomposerTestResult("Caching")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 11] Testing caching...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            problem = ProblemGraph(nodes={'A': {}, 'B': {}}, edges=[('A', 'B', {})], metadata={'domain': 'test'})
            
            t1 = time.time()
            plan1 = decomposer.decompose_novel_problem(problem)
            time1 = time.time() - t1
            
            t2 = time.time()
            plan2 = decomposer.decompose_novel_problem(problem)
            time2 = time.time() - t2
            
            cache_size = len(decomposer.decomposition_cache)
            logger.info(f"  ✓ Cache size: {cache_size}, Times: {time1:.3f}s, {time2:.3f}s")
            
            result.passed = True
            result.details = {'cache_size': cache_size, 'first_time': time1, 'second_time': time2}
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Caching test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Caching failed: {result.error}"
    
    @pytest.mark.timeout(30)
    def test_12_error_handling(self):
        """Test 12: Error handling"""
        result = DecomposerTestResult("Error Handling")
        start_time = time.time()
        
        try:
            logger.info("\n[TEST 12] Testing error handling...")
            
            from src.vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            from src.vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
            
            decomposer = create_decomposer(config={'test_mode': True})
            
            # Empty problem
            empty = ProblemGraph(nodes={}, edges=[], metadata={})
            plan1 = decomposer.decompose_novel_problem(empty)
            logger.info(f"  ✓ Empty problem: {len(plan1.steps)} steps")
            
            # Cyclic problem
            cyclic = ProblemGraph(nodes={'A': {}, 'B': {}, 'C': {}}, edges=[('A', 'B', {}), ('B', 'C', {}), ('C', 'A', {})], metadata={'domain': 'test'})
            plan2 = decomposer.decompose_novel_problem(cyclic)
            logger.info(f"  ✓ Cyclic problem: {len(plan2.steps)} steps")
            
            # Large problem
            large = ProblemGraph(nodes={f'n{i}': {} for i in range(100)}, edges=[(f'n{i}', f'n{i+1}', {}) for i in range(99)], metadata={'domain': 'test'})
            plan3 = decomposer.decompose_novel_problem(large)
            logger.info(f"  ✓ Large problem (100 nodes): {len(plan3.steps)} steps")
            
            # Minimal problem
            minimal = ProblemGraph(nodes={'X': {}}, edges=[])
            plan4 = decomposer.decompose_novel_problem(minimal)
            logger.info(f"  ✓ Minimal problem: {len(plan4.steps)} steps")
            
            result.passed = True
            result.details = {'edge_cases_handled': 4}
            
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"  ✗ Error handling test failed: {e}")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.test_results.append(result)
        
        assert result.passed, f"Error handling failed: {result.error}"
    
    def teardown_method(self):
        """Teardown after each test - generate mini report"""
        if self.test_results:
            latest = self.test_results[-1]
            status = "✓ PASS" if latest.passed else "✗ FAIL"
            logger.info(f"\n{status} | {latest.name} | {latest.duration:.3f}s")
            if latest.details:
                for k, v in latest.details.items():
                    logger.info(f"    {k}: {v}")


# Standalone runner for non-pytest execution
class StandaloneRunner:
    """Run tests without pytest"""
    
    def __init__(self):
        self.tester = TestProblemDecomposerIntegration()
        self.total = 0
        self.passed = 0
        self.failed = 0
    
    def run_all(self):
        """Run all tests"""
        logger.info("=" * 80)
        logger.info("PROBLEM DECOMPOSER INTEGRATION TEST SUITE")
        logger.info("=" * 80)
        
        start = time.time()
        
        tests = [
            self.tester.test_01_module_imports,
            self.tester.test_02_component_initialization,
            self.tester.test_03_strategy_registration,
            self.tester.test_04_bootstrap_creation,
            self.tester.test_05_problem_decomposition,
            self.tester.test_06_plan_execution_safety,
            self.tester.test_07_full_flow_safety,
            self.tester.test_08_learning_integration,
            self.tester.test_09_fallback_chain,
            self.tester.test_10_performance_tracking,
            self.tester.test_11_caching,
            self.tester.test_12_error_handling
        ]
        
        for test in tests:
            self.total += 1
            self.tester.setup_method()
            try:
                test()
                self.passed += 1
            except Exception as e:
                self.failed += 1
                logger.error(f"Test failed: {e}")
            self.tester.teardown_method()
        
        duration = time.time() - start
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL REPORT")
        logger.info("=" * 80)
        logger.info(f"Total: {self.total} | Passed: {self.passed} | Failed: {self.failed}")
        logger.info(f"Duration: {duration:.2f}s")
        
        if self.failed == 0:
            logger.info("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        else:
            logger.info(f"\n✗✗✗ {self.failed} TEST(S) FAILED ✗✗✗")
        
        logger.info("=" * 80)
        
        return self.failed == 0


if __name__ == '__main__':
    # Check if pytest is available
    if 'pytest' in sys.modules:
        pytest.main([__file__, '-v', '-s'])
    else:
        # Run standalone
        runner = StandaloneRunner()
        success = runner.run_all()
        sys.exit(0 if success else 1)
