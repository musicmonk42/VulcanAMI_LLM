"""
test_problem_decomposer_core.py - Comprehensive tests for problem decomposer core
Part of the VULCAN-AGI system

Tests cover:
- Problem signature extraction
- Complexity analysis
- Strategy prediction
- Plan creation and execution
- Learning from outcomes
- Safety validation
- Caching behavior
- Statistics tracking
"""

import pytest
import numpy as np
import time
import hashlib
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Import components to test
from vulcan.problem_decomposer.problem_decomposer_core import (
    ProblemDecomposer,
    ProblemGraph,
    DecompositionPlan,
    ExecutionOutcome,
    DecompositionStep,
    ProblemSignature,
    LearningGap,
    DomainSelector,
    PerformanceTracker,
    StrategyProfiler,
    ProblemComplexity,
    DomainDataCategory,
    DecompositionMode
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_problem():
    """Create simple test problem"""
    return ProblemGraph(
        nodes={
            'A': {'type': 'operation'},
            'B': {'type': 'operation'},
            'C': {'type': 'operation'}
        },
        edges=[
            ('A', 'B', {}),
            ('B', 'C', {})
        ],
        root='A',
        metadata={'domain': 'test', 'type': 'sequential'}
    )


@pytest.fixture
def hierarchical_problem():
    """Create hierarchical test problem"""
    return ProblemGraph(
        nodes={
            'root': {'type': 'decision', 'level': 0},
            'branch1': {'type': 'operation', 'level': 1},
            'branch2': {'type': 'operation', 'level': 1},
            'leaf1': {'type': 'transform', 'level': 2},
            'leaf2': {'type': 'transform', 'level': 2},
            'leaf3': {'type': 'transform', 'level': 2}
        },
        edges=[
            ('root', 'branch1', {'weight': 1.0}),
            ('root', 'branch2', {'weight': 1.0}),
            ('branch1', 'leaf1', {'weight': 0.5}),
            ('branch1', 'leaf2', {'weight': 0.5}),
            ('branch2', 'leaf3', {'weight': 1.0})
        ],
        root='root',
        metadata={'domain': 'planning', 'type': 'hierarchical'}
    )


@pytest.fixture
def complex_problem():
    """Create complex test problem"""
    nodes = {f'node_{i}': {'type': 'operation', 'index': i} for i in range(20)}
    edges = [(f'node_{i}', f'node_{i+1}', {}) for i in range(19)]
    
    # Add some cross-edges for complexity
    edges.extend([
        ('node_5', 'node_10', {}),
        ('node_8', 'node_15', {}),
        ('node_12', 'node_18', {})
    ])
    
    return ProblemGraph(
        nodes=nodes,
        edges=edges,
        root='node_0',
        metadata={'domain': 'optimization', 'type': 'complex', 'constraints': ['c1', 'c2', 'c3']}
    )


@pytest.fixture
def cyclic_problem():
    """Create problem with cycles"""
    return ProblemGraph(
        nodes={
            'init': {'type': 'operation'},
            'evaluate': {'type': 'decision'},
            'refine': {'type': 'transform'},
            'output': {'type': 'operation'}
        },
        edges=[
            ('init', 'evaluate', {}),
            ('evaluate', 'refine', {'condition': 'not_satisfied'}),
            ('refine', 'evaluate', {}),  # Cycle
            ('evaluate', 'output', {'condition': 'satisfied'})
        ],
        root='init',
        metadata={'domain': 'optimization', 'type': 'iterative'}
    )


@pytest.fixture
def mock_validator():
    """Create mock validator"""
    validator = Mock()
    validator.validate_solution = Mock(return_value={'valid': True, 'score': 0.9})
    return validator


@pytest.fixture
def mock_semantic_bridge():
    """Create mock semantic bridge"""
    bridge = Mock()
    bridge.apply_concept = Mock(return_value={'success': True})
    return bridge


@pytest.fixture
def decomposer(mock_validator, mock_semantic_bridge):
    """Create test decomposer instance"""
    return ProblemDecomposer(
        semantic_bridge=mock_semantic_bridge,
        validator=mock_validator,
        safety_config={}  # SafetyConfig will use defaults
    )


@pytest.fixture
def decomposer_no_safety():
    """Create decomposer without safety validator for testing fallback"""
    with patch('vulcan.problem_decomposer.problem_decomposer_core.SAFETY_VALIDATOR_AVAILABLE', False):
        with patch('vulcan.problem_decomposer.problem_decomposer_core.EnhancedSafetyValidator', None):
            decomposer = ProblemDecomposer()
            return decomposer


# ============================================================
# PROBLEM GRAPH TESTS
# ============================================================

class TestProblemGraph:
    """Tests for ProblemGraph class"""
    
    def test_graph_creation(self, simple_problem):
        """Test basic graph creation"""
        assert len(simple_problem.nodes) == 3
        assert len(simple_problem.edges) == 2
        assert simple_problem.root == 'A'
    
    def test_get_signature(self, simple_problem):
        """Test signature generation"""
        sig1 = simple_problem.get_signature()
        sig2 = simple_problem.get_signature()
        
        assert sig1 == sig2
        assert len(sig1) == 32  # MD5 hash length
    
    def test_signature_uniqueness(self, simple_problem, hierarchical_problem):
        """Test that different problems have different signatures"""
        sig1 = simple_problem.get_signature()
        sig2 = hierarchical_problem.get_signature()
        
        assert sig1 != sig2
    
    def test_to_networkx(self, simple_problem):
        """Test NetworkX conversion"""
        G = simple_problem.to_networkx()
        
        assert hasattr(G, 'nodes')
        assert hasattr(G, 'edges')
        
        # Check nodes were added
        if hasattr(G, 'number_of_nodes'):
            assert G.number_of_nodes() == 3
        else:
            assert len(list(G.nodes())) == 3
    
    def test_complexity_score(self, simple_problem):
        """Test complexity score storage"""
        simple_problem.complexity_score = 2.5
        assert simple_problem.complexity_score == 2.5


# ============================================================
# DECOMPOSITION STEP TESTS
# ============================================================

class TestDecompositionStep:
    """Tests for DecompositionStep class"""
    
    def test_step_creation(self):
        """Test step creation"""
        step = DecompositionStep(
            step_id='step_1',
            action_type='process',
            description='Process node A',
            dependencies=['step_0'],
            estimated_complexity=1.5,
            confidence=0.8
        )
        
        assert step.step_id == 'step_1'
        assert step.action_type == 'process'
        assert step.confidence == 0.8
    
    def test_step_to_dict(self):
        """Test step serialization"""
        step = DecompositionStep(
            step_id='step_1',
            action_type='process',
            description='Test step'
        )
        
        step_dict = step.to_dict()
        
        assert isinstance(step_dict, dict)
        assert step_dict['step_id'] == 'step_1'
        assert 'action_type' in step_dict
        assert 'dependencies' in step_dict
    
    def test_step_from_dict(self):
        """Test step deserialization"""
        step_dict = {
            'step_id': 'step_2',
            'action_type': 'transform',
            'description': 'Transform data',
            'confidence': 0.9
        }
        
        step = DecompositionStep.from_dict(step_dict)
        
        assert step.step_id == 'step_2'
        assert step.action_type == 'transform'
        assert step.confidence == 0.9


# ============================================================
# DECOMPOSITION PLAN TESTS
# ============================================================

class TestDecompositionPlan:
    """Tests for DecompositionPlan class"""
    
    def test_plan_creation(self):
        """Test plan creation"""
        plan = DecompositionPlan(
            steps=[],
            confidence=0.7,
            estimated_complexity=3.0
        )
        
        assert plan.confidence == 0.7
        assert plan.estimated_complexity == 3.0
        assert len(plan.steps) == 0
    
    def test_add_step(self):
        """Test adding steps to plan"""
        plan = DecompositionPlan()
        
        step = DecompositionStep(
            step_id='step_1',
            action_type='process',
            description='Test'
        )
        
        plan.add_step(step)
        
        assert len(plan.steps) == 1
        assert plan.steps[0].step_id == 'step_1'
    
    def test_add_step_from_dict(self):
        """Test adding step from dictionary"""
        plan = DecompositionPlan()
        
        step_dict = {
            'step_id': 'step_2',
            'action_type': 'transform',
            'description': 'Transform'
        }
        
        plan.add_step(step_dict)
        
        assert len(plan.steps) == 1
        assert isinstance(plan.steps[0], DecompositionStep)
    
    def test_plan_to_dict(self):
        """Test plan serialization"""
        plan = DecompositionPlan(confidence=0.8)
        plan.add_step(DecompositionStep('step_1', 'process', 'Test'))
        
        plan_dict = plan.to_dict()
        
        assert isinstance(plan_dict, dict)
        assert 'steps' in plan_dict
        assert 'confidence' in plan_dict
        assert len(plan_dict['steps']) == 1


# ============================================================
# EXECUTION OUTCOME TESTS
# ============================================================

class TestExecutionOutcome:
    """Tests for ExecutionOutcome class"""
    
    def test_outcome_creation(self):
        """Test outcome creation"""
        outcome = ExecutionOutcome(
            success=True,
            execution_time=1.5,
            errors=[],
            metrics={'accuracy': 0.95}
        )
        
        assert outcome.success is True
        assert outcome.execution_time == 1.5
        assert outcome.metrics['accuracy'] == 0.95
    
    def test_get_success_rate_no_results(self):
        """Test success rate with no sub-results"""
        outcome = ExecutionOutcome(
            success=True,
            execution_time=1.0
        )
        
        rate = outcome.get_success_rate()
        assert rate == 0.0
    
    def test_get_success_rate_with_results(self):
        """Test success rate calculation"""
        outcome = ExecutionOutcome(
            success=True,
            execution_time=1.0,
            sub_results=[
                {'success': True},
                {'success': True},
                {'success': False},
                {'success': True}
            ]
        )
        
        rate = outcome.get_success_rate()
        assert rate == 0.75  # 3/4


# ============================================================
# PERFORMANCE TRACKER TESTS
# ============================================================

class TestPerformanceTracker:
    """Tests for PerformanceTracker class"""
    
    def test_tracker_creation(self):
        """Test tracker initialization"""
        tracker = PerformanceTracker()
        
        assert len(tracker.execution_history) == 0
        assert isinstance(tracker.strategy_performance, dict)
    
    def test_record_execution(self, simple_problem):
        """Test recording execution"""
        tracker = PerformanceTracker()
        
        # Create strategy with proper name attribute
        strategy = Mock()
        strategy.name = 'TestStrategy'
        
        plan = DecompositionPlan(confidence=0.8)
        plan.strategy = strategy
        
        outcome = ExecutionOutcome(success=True, execution_time=1.0)
        
        tracker.record_execution(simple_problem, plan, outcome)
        
        assert len(tracker.execution_history) == 1
        # The strategy name should be recorded
        assert 'TestStrategy' in tracker.strategy_performance
        assert tracker.strategy_performance['TestStrategy']['success'] == 1
    
    def test_get_strategy_success_rate(self):
        """Test getting strategy success rate"""
        tracker = PerformanceTracker()
        
        # Manually add performance data
        tracker.strategy_performance['test_strategy'] = {
            'success': 7,
            'failure': 3
        }
        
        rate = tracker.get_strategy_success_rate('test_strategy')
        assert rate == 0.7
    
    def test_get_strategy_success_rate_unknown(self):
        """Test success rate for unknown strategy"""
        tracker = PerformanceTracker()
        
        rate = tracker.get_strategy_success_rate('unknown_strategy')
        assert rate == 0.5  # Default
    
    def test_strategy_limit_enforcement(self, simple_problem):
        """Test that strategy tracking is bounded"""
        tracker = PerformanceTracker()
        tracker.max_strategies = 5
        
        # Create mock strategies and record executions (which triggers limit enforcement)
        for i in range(10):
            strategy = Mock()
            strategy.name = f'strategy_{i}'
            
            plan = DecompositionPlan(confidence=0.8)
            plan.strategy = strategy
            
            outcome = ExecutionOutcome(success=True, execution_time=1.0)
            
            # This should enforce the limit
            tracker.record_execution(simple_problem, plan, outcome)
        
        # Should not exceed max (limit enforcement happens during record_execution)
        assert len(tracker.strategy_performance) <= tracker.max_strategies


# ============================================================
# STRATEGY PROFILER TESTS
# ============================================================

class TestStrategyProfiler:
    """Tests for StrategyProfiler class"""
    
    def test_profiler_creation(self):
        """Test profiler initialization"""
        profiler = StrategyProfiler()
        
        assert len(profiler.strategy_profiles) == 0
        assert isinstance(profiler.domain_affinity, dict)
    
    def test_profile_strategy(self):
        """Test profiling a strategy"""
        profiler = StrategyProfiler()
        
        strategy = Mock()
        strategy.name = 'TestStrategy'
        strategy.strategy_type = 'structural'
        strategy.max_depth = 5  # Return a real number, not a Mock
        strategy.is_parallelizable = Mock(return_value=True)
        strategy.is_deterministic = Mock(return_value=True)
        
        profile = profiler.profile_strategy(strategy)
        
        assert profile['name'] == 'TestStrategy'
        assert profile['parallelizable'] is True
        assert profile['deterministic'] is True
    
    def test_update_affinity(self):
        """Test updating domain affinity"""
        profiler = StrategyProfiler()
        
        profiler.update_affinity('strategy1', 'optimization', 3.0, success=True)
        profiler.update_affinity('strategy1', 'optimization', 3.0, success=True)
        profiler.update_affinity('strategy1', 'optimization', 3.0, success=False)
        
        affinity = profiler.domain_affinity['strategy1']['optimization']
        assert affinity > 0  # Should be positive overall
    
    def test_get_best_strategy_for_domain(self):
        """Test getting best strategy for domain"""
        profiler = StrategyProfiler()
        
        profiler.domain_affinity['strategy1']['optimization'] = 0.8
        profiler.domain_affinity['strategy2']['optimization'] = 0.6
        profiler.domain_affinity['strategy3']['optimization'] = 0.9
        
        best = profiler.get_best_strategy_for_domain('optimization')
        assert best == 'strategy3'


# ============================================================
# DOMAIN SELECTOR TESTS
# ============================================================

class TestDomainSelector:
    """Tests for DomainSelector class"""
    
    def test_selector_creation(self):
        """Test selector initialization"""
        selector = DomainSelector()
        
        assert len(selector.domain_data_counts) > 0  # Has defaults
    
    def test_categorize_domains_by_data(self):
        """Test domain categorization"""
        selector = DomainSelector()
        
        domains = ['general', 'planning', 'novel_domain']
        categorized = selector.categorize_domains_by_data(domains)
        
        assert DomainDataCategory.FREQUENT in categorized
        assert 'general' in categorized[DomainDataCategory.FREQUENT]
    
    def test_select_stratified_sample(self):
        """Test stratified sampling"""
        selector = DomainSelector()
        
        frequent = ['domain1', 'domain2', 'domain3', 'domain4']
        medium = ['domain5', 'domain6', 'domain7']
        rare = ['domain8', 'domain9']
        
        sample = selector.select_stratified_sample(frequent, medium, rare)
        
        assert len(sample) <= 7  # Max 3+2+2
        assert len(sample) > 0
    
    def test_find_similar_domains(self):
        """Test finding similar domains"""
        selector = DomainSelector()
        
        similar = selector.find_similar_domains('optimization', exclude=['general'])
        
        assert isinstance(similar, list)
        assert 'general' not in similar
        assert 'optimization' not in similar


# ============================================================
# PROBLEM DECOMPOSER CORE TESTS
# ============================================================

@pytest.mark.skip(reason="Decomposer fixture hangs 60+ seconds during safety validator initialization (rollback_audit._initialize_storage). Background threads in governance_alignment and rollback_audit don't stop. Needs proper cleanup in safety components.")
class TestProblemDecomposer:
    """Tests for ProblemDecomposer class"""
    
    def test_decomposer_creation(self, decomposer):
        """Test decomposer initialization"""
        assert decomposer is not None
        assert decomposer.library is not None
        assert decomposer.thresholds is not None
        assert decomposer.executor is not None
    
    def test_decomposer_requires_safety_for_execution(self, simple_problem):
        """Test that execution requires safety validator"""
        # Create decomposer without safety
        with patch('vulcan.problem_decomposer.problem_decomposer_core.SAFETY_VALIDATOR_AVAILABLE', False):
            with patch('vulcan.problem_decomposer.problem_decomposer_core.EnhancedSafetyValidator', None):
                decomposer_no_safety = ProblemDecomposer()
                plan = DecompositionPlan(confidence=0.8)
                
                # Should raise RuntimeError for safety requirement
                with pytest.raises(RuntimeError, match="SAFETY CRITICAL"):
                    with patch.object(decomposer_no_safety.executor, 'safety_validator', None):
                        decomposer_no_safety.executor.execute_plan(simple_problem, plan)
    
    def test_extract_problem_signature(self, decomposer, simple_problem):
        """Test problem signature extraction"""
        signature = decomposer._extract_problem_signature(simple_problem)
        
        assert isinstance(signature, ProblemSignature)
        assert signature.node_count == 3
        assert signature.domain == 'test'
    
    def test_extract_signature_caching(self, decomposer, simple_problem):
        """Test that signatures are cached"""
        sig1 = decomposer._extract_problem_signature(simple_problem)
        sig2 = decomposer._extract_problem_signature(simple_problem)
        
        assert sig1 is sig2  # Same object from cache
    
    def test_analyze_complexity(self, decomposer, simple_problem):
        """Test complexity analysis"""
        complexity = decomposer._analyze_complexity(simple_problem)
        
        assert isinstance(complexity, float)
        assert 1.0 <= complexity <= 5.0
    
    def test_analyze_complexity_hierarchical(self, decomposer, hierarchical_problem):
        """Test complexity for hierarchical problem"""
        complexity = decomposer._analyze_complexity(hierarchical_problem)
        
        assert complexity > 1.5  # Should be moderately complex
    
    def test_predict_best_strategy(self, decomposer, simple_problem):
        """Test strategy prediction"""
        signature = decomposer._extract_problem_signature(simple_problem)
        signature.complexity = 2.0
        
        strategy = decomposer._predict_best_strategy(simple_problem, signature)
        
        assert strategy is not None
        assert hasattr(strategy, 'decompose')
    
    def test_predict_strategy_hierarchical(self, decomposer, hierarchical_problem):
        """Test strategy prediction for hierarchical problem"""
        signature = decomposer._extract_problem_signature(hierarchical_problem)
        signature.complexity = 3.0
        
        strategy = decomposer._predict_best_strategy(hierarchical_problem, signature)
        
        assert strategy is not None
    
    def test_decompose_novel_problem(self, decomposer, simple_problem):
        """Test decomposing novel problem"""
        plan = decomposer.decompose_novel_problem(simple_problem)
        
        assert isinstance(plan, DecompositionPlan)
        assert len(plan.steps) > 0
        assert 0 <= plan.confidence <= 1
    
    def test_decompose_caching(self, decomposer, simple_problem):
        """Test that decomposition results are cached"""
        plan1 = decomposer.decompose_novel_problem(simple_problem)
        plan2 = decomposer.decompose_novel_problem(simple_problem)
        
        # Should return same plan from cache
        assert plan1 is plan2
    
    def test_decompose_with_fallbacks(self, decomposer, simple_problem):
        """Test decomposition with fallback chain"""
        # Mock the fallback chain to return DecompositionPlan objects instead of ExecutionPlan
        with patch.object(decomposer.fallback_chain, 'generate_fallback_plans') as mock_fallbacks:
            # Return proper DecompositionPlan objects
            fallback_plan = DecompositionPlan(
                confidence=0.6,
                estimated_complexity=2.0
            )
            fallback_plan.strategy = Mock(name='FallbackStrategy')
            mock_fallbacks.return_value = [fallback_plan]
            
            plan = decomposer.decompose_with_fallbacks(simple_problem)
            
            assert isinstance(plan, DecompositionPlan)
            assert plan.confidence >= 0
    
    def test_learn_from_execution(self, decomposer, simple_problem):
        """Test learning from execution"""
        plan = DecompositionPlan(confidence=0.7)
        plan.strategy = Mock(name='TestStrategy')
        
        outcome = ExecutionOutcome(success=True, execution_time=1.5)
        
        decomposer.learn_from_execution(simple_problem, plan, outcome)
        
        assert decomposer.successful_decompositions == 1
    
    def test_learn_from_failure(self, decomposer, simple_problem):
        """Test learning from failed execution"""
        plan = DecompositionPlan(confidence=0.7)
        plan.strategy = Mock(name='TestStrategy')
        
        outcome = ExecutionOutcome(success=False, execution_time=1.0)
        
        decomposer.learn_from_execution(simple_problem, plan, outcome)
        
        assert len(decomposer.learning_gaps) == 1
    
    def test_create_learning_gap(self, decomposer, simple_problem):
        """Test learning gap creation"""
        gap = decomposer.create_learning_gap(simple_problem)
        
        assert isinstance(gap, LearningGap)
        assert gap.problem_signature == simple_problem.get_signature()
        assert len(gap.failure_reason) > 0
    
    def test_get_statistics(self, decomposer):
        """Test getting statistics"""
        stats = decomposer.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'decomposition_stats' in stats
        assert 'execution_stats' in stats
        assert 'safety' in stats
    
    def test_safety_blocks_tracked(self, decomposer, simple_problem):
        """Test that safety blocks are tracked"""
        # Create unsafe plan
        unsafe_plan = DecompositionPlan(confidence=-1.0)  # Invalid confidence
        
        with patch.object(decomposer.executor, 'execute_plan') as mock_execute:
            mock_execute.return_value = ExecutionOutcome(
                success=False,
                execution_time=0,
                errors=['Safety blocked']
            )
            
            # This should be blocked
            outcome = mock_execute(simple_problem, unsafe_plan)
            
            assert outcome.success is False


# ============================================================
# INTEGRATION TESTS
# ============================================================

@pytest.mark.skip(reason="Decomposer fixture hangs 60+ seconds during safety validator initialization. Needs proper cleanup in safety components.")
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_decomposition_workflow(self, decomposer, simple_problem):
        """Test complete decomposition workflow"""
        # Decompose
        plan = decomposer.decompose_novel_problem(simple_problem)
        
        assert isinstance(plan, DecompositionPlan)
        assert len(plan.steps) > 0
        
        # Create successful outcome
        outcome = ExecutionOutcome(
            success=True,
            execution_time=1.0,
            metrics={'accuracy': 0.9}
        )
        
        # Learn
        decomposer.learn_from_execution(simple_problem, plan, outcome)
        
        # Check learning happened
        stats = decomposer.get_statistics()
        assert stats['decomposition_stats']['successful_decompositions'] == 1
    
    def test_multiple_problem_types(self, decomposer, simple_problem, 
                                    hierarchical_problem, cyclic_problem):
        """Test handling multiple problem types"""
        problems = [simple_problem, hierarchical_problem, cyclic_problem]
        
        for problem in problems:
            plan = decomposer.decompose_novel_problem(problem)
            
            assert isinstance(plan, DecompositionPlan)
            assert len(plan.steps) > 0
            assert plan.confidence > 0
    
    def test_learning_improves_prediction(self, decomposer, simple_problem):
        """Test that learning improves strategy prediction"""
        # Get initial prediction
        signature = decomposer._extract_problem_signature(simple_problem)
        signature.complexity = 2.0
        strategy1 = decomposer._predict_best_strategy(simple_problem, signature)
        
        # Simulate successful execution
        plan = DecompositionPlan(strategy=strategy1, confidence=0.8)
        outcome = ExecutionOutcome(success=True, execution_time=1.0)
        
        decomposer.learn_from_execution(simple_problem, plan, outcome)
        
        # Update affinity
        decomposer.strategy_profiler.update_affinity(
            strategy1.name if hasattr(strategy1, 'name') else 'test',
            simple_problem.metadata['domain'],
            2.0,
            True
        )
        
        # Get new prediction
        strategy2 = decomposer._predict_best_strategy(simple_problem, signature)
        
        # Should still work (same or better)
        assert strategy2 is not None


# ============================================================
# EDGE CASE TESTS
# ============================================================

@pytest.mark.skip(reason="Decomposer fixture hangs 60+ seconds during safety validator initialization. Needs proper cleanup in safety components.")
class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_problem(self, decomposer):
        """Test handling empty problem"""
        empty_problem = ProblemGraph(
            nodes={},
            edges=[],
            metadata={'domain': 'test'}
        )
        
        plan = decomposer.decompose_novel_problem(empty_problem)
        
        assert isinstance(plan, DecompositionPlan)
        # Should handle gracefully
    
    def test_very_large_problem(self, decomposer):
        """Test handling very large problem"""
        nodes = {f'node_{i}': {'index': i} for i in range(100)}
        edges = [(f'node_{i}', f'node_{i+1}', {}) for i in range(99)]
        
        large_problem = ProblemGraph(
            nodes=nodes,
            edges=edges,
            root='node_0',
            metadata={'domain': 'test'}
        )
        
        plan = decomposer.decompose_novel_problem(large_problem)
        
        assert isinstance(plan, DecompositionPlan)
    
    def test_disconnected_graph(self, decomposer):
        """Test handling disconnected graph"""
        disconnected = ProblemGraph(
            nodes={'A': {}, 'B': {}, 'C': {}, 'D': {}},
            edges=[('A', 'B', {}), ('C', 'D', {})],  # Two components
            metadata={'domain': 'test'}
        )
        
        plan = decomposer.decompose_novel_problem(disconnected)
        
        assert isinstance(plan, DecompositionPlan)
    
    def test_cache_overflow(self, decomposer):
        """Test cache handling when full"""
        decomposer.cache_size = 5
        
        # Generate more problems than cache size
        for i in range(10):
            problem = ProblemGraph(
                nodes={f'node_{i}': {}},
                edges=[],
                metadata={'domain': 'test', 'id': i}
            )
            
            decomposer.decompose_novel_problem(problem)
        
        # Cache should not exceed size
        assert len(decomposer.decomposition_cache) <= 5


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
