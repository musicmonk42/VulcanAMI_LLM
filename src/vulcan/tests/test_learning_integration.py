"""
test_learning_integration.py - Comprehensive tests for learning integration
Part of the VULCAN-AGI system

Tests:
- Problem to experience conversion
- Difficulty estimation
- RLHF feedback routing
- Integrated learning coordination
- Unified decomposer learner
- Principle learning integration
- State persistence
"""

import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for learning_integration tests")

import numpy as np
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test
from problem_decomposer.learning_integration import (
    ProblemToExperienceConverter,
    DecompositionDifficultyEstimator,
    RLHFFeedbackRouter,
    IntegratedLearningCoordinator,
    UnifiedDecomposerLearner,
    create_unified_decomposer
)

from problem_decomposer.problem_decomposer_core import (
    ProblemGraph,
    DecompositionPlan,
    ExecutionOutcome,
    DecompositionStep
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_problem():
    """Create simple problem graph"""
    problem = ProblemGraph(
        nodes={
            'A': {'type': 'operation', 'complexity': 1.0},
            'B': {'type': 'operation', 'complexity': 1.5},
            'C': {'type': 'result', 'complexity': 0.5}
        },
        edges=[
            ('A', 'B', {'weight': 1.0}),
            ('B', 'C', {'weight': 0.5})
        ],
        root='A',
        metadata={'domain': 'test', 'type': 'simple'}
    )
    problem.complexity_score = 2.0
    return problem


@pytest.fixture
def complex_problem():
    """Create complex problem graph"""
    nodes = {f'node_{i}': {'type': 'operation', 'complexity': 1.0} for i in range(20)}
    edges = [(f'node_{i}', f'node_{i+1}', {}) for i in range(19)]
    
    problem = ProblemGraph(
        nodes=nodes,
        edges=edges,
        root='node_0',
        metadata={
            'domain': 'complex_planning',
            'constraints': ['c1', 'c2', 'c3']
        }
    )
    problem.complexity_score = 5.0
    return problem


@pytest.fixture
def sample_plan():
    """Create sample decomposition plan"""
    # Create plan with DecompositionStep objects
    step1 = DecompositionStep(
        step_id='step_1',
        action_type='process',
        description='First step',
        dependencies=[],
        estimated_complexity=1.0
    )
    
    step2 = DecompositionStep(
        step_id='step_2',
        action_type='process',
        description='Second step',
        dependencies=['step_1'],
        estimated_complexity=1.5
    )
    
    plan = DecompositionPlan(
        steps=[step1, step2],
        estimated_complexity=2.5,
        confidence=0.8
    )
    
    # Add mock strategy
    plan.strategy = Mock()
    plan.strategy.name = "TestStrategy"
    
    return plan


@pytest.fixture
def successful_outcome():
    """Create successful execution outcome"""
    return ExecutionOutcome(
        success=True,
        execution_time=1.5,
        sub_results=[
            {'step': 'step_1', 'success': True, 'time': 0.5},
            {'step': 'step_2', 'success': True, 'time': 1.0}
        ],
        metrics={
            'actual_complexity': 2.0,
            'solution_quality': 0.9
        },
        errors=[]
    )


@pytest.fixture
def failed_outcome():
    """Create failed execution outcome"""
    return ExecutionOutcome(
        success=False,
        execution_time=0.5,
        sub_results=[
            {'step': 'step_1', 'success': True, 'time': 0.3},
            {'step': 'step_2', 'success': False, 'time': 0.2}
        ],
        metrics={
            'actual_complexity': 3.0
        },
        errors=['Step 2 failed: timeout', 'Resource exceeded']
    )


@pytest.fixture
def mock_decomposer():
    """Create mock decomposer"""
    decomposer = Mock()
    decomposer.get_statistics = Mock(return_value={
        'decomposition_stats': {
            'success_rate': 0.75,
            'total_decompositions': 100
        }
    })
    decomposer.learn_from_execution = Mock()
    decomposer.strategy_library = Mock()
    return decomposer


@pytest.fixture
def mock_continual_learner():
    """Create mock continual learner"""
    learner = Mock()
    learner.process_experience = Mock(return_value={'task_id': 'task_1'})
    learner.get_statistics = Mock(return_value={
        'continual_metrics': {'forgetting_measure': 0.1},
        'num_tasks': 5,
        'free_capacity': 0.5
    })
    return learner


@pytest.fixture
def mock_curriculum_learner():
    """Create mock curriculum learner"""
    learner = Mock()
    learner.curriculum_stages = [1, 2, 3]
    learner.generate_curriculum = Mock()
    learner.get_next_batch = Mock(return_value=[])
    learner.get_curriculum_stats = Mock(return_value={
        'difficulty_adjustments': [0.5, 0.6, 0.7]
    })
    learner.difficulty_estimator = None
    return learner


@pytest.fixture
def mock_meta_learner():
    """Create mock meta learner"""
    learner = Mock()
    learner.online_meta_update = Mock()
    learner.get_statistics = Mock(return_value={
        'meta_updates': 10
    })
    return learner


@pytest.fixture
def mock_metacognition():
    """Create mock metacognition"""
    monitor = Mock()
    monitor.update_self_model = Mock()
    monitor.analyze_learning_efficiency = Mock(return_value={
        'recommendations': [{'suggestion': 'Test recommendation'}]
    })
    return monitor


@pytest.fixture
def mock_rlhf_manager():
    """Create mock RLHF manager"""
    manager = Mock()
    manager.receive_feedback = Mock()
    manager.get_statistics = Mock(return_value={
        'total_feedback': 50
    })
    return manager


@pytest.fixture
def mock_principle_learner():
    """Create mock principle learner"""
    learner = Mock()
    learner.extract_and_promote = Mock(return_value={
        'principles_extracted': 2,
        'principles_validated': 1,
        'principles_promoted': 1
    })
    learner.find_applicable_principles = Mock(return_value=[])
    learner.prune_low_quality_principles = Mock(return_value=5)
    learner.get_learning_statistics = Mock(return_value={
        'promotion': {
            'promoter_stats': {'promotion_rate': 0.5}
        },
        'knowledge_base': {'total_principles': 50}
    })
    learner.knowledge_base = Mock()
    learner.knowledge_base.export = Mock(return_value={'path': 'test.json'})
    learner.knowledge_base.import_from = Mock()
    return learner


@pytest.fixture
def temp_directory():
    """Create temporary directory for file operations"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================
# PROBLEM TO EXPERIENCE CONVERTER TESTS
# ============================================================

class TestProblemToExperienceConverter:
    """Test ProblemToExperienceConverter"""
    
    def test_converter_initialization(self):
        """Test converter initialization"""
        converter = ProblemToExperienceConverter(embedding_dim=256)
        
        assert converter.embedding_dim == 256
        assert len(converter.problem_embeddings) == 0
        
        logger.info("✓ Converter initialization test passed")
    
    def test_problem_to_embedding(self, simple_problem):
        """Test problem to embedding conversion"""
        converter = ProblemToExperienceConverter()
        
        embedding = converter.problem_to_embedding(simple_problem)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == converter.embedding_dim
        assert embedding.dtype == np.float32
        
        logger.info("✓ Problem to embedding test passed")
    
    def test_embedding_caching(self, simple_problem):
        """Test embedding caching"""
        converter = ProblemToExperienceConverter()
        
        # First call
        embedding1 = converter.problem_to_embedding(simple_problem)
        
        # Second call should use cache
        embedding2 = converter.problem_to_embedding(simple_problem)
        
        # Should be exactly the same object (cached)
        assert np.array_equal(embedding1, embedding2)
        assert len(converter.problem_embeddings) == 1
        
        logger.info("✓ Embedding caching test passed")
    
    def test_cache_size_limit(self):
        """Test cache size enforcement"""
        converter = ProblemToExperienceConverter()
        converter.max_cache_size = 10
        
        # Create many different problems
        for i in range(15):
            problem = ProblemGraph(
                nodes={f'node_{i}': {}},
                edges=[],
                metadata={'id': i}
            )
            problem.complexity_score = float(i)
            converter.problem_to_embedding(problem)
        
        # Cache should be limited
        assert len(converter.problem_embeddings) <= converter.max_cache_size
        
        logger.info("✓ Cache size limit test passed")
    
    def test_convert_to_experience_success(self, simple_problem, sample_plan, successful_outcome):
        """Test converting successful execution to experience"""
        converter = ProblemToExperienceConverter()
        
        experience = converter.convert_to_experience(simple_problem, sample_plan, successful_outcome)
        
        assert 'embedding' in experience
        assert 'reward' in experience
        assert 'loss' in experience
        assert 'modality' in experience
        assert experience['modality'] == 'problem_decomposition'
        assert experience['reward'] > 0  # Successful outcome should have positive reward
        
        logger.info("✓ Convert to experience (success) test passed")
    
    def test_convert_to_experience_failure(self, simple_problem, sample_plan, failed_outcome):
        """Test converting failed execution to experience"""
        converter = ProblemToExperienceConverter()
        
        experience = converter.convert_to_experience(simple_problem, sample_plan, failed_outcome)
        
        assert 'embedding' in experience
        assert 'reward' in experience
        assert 'loss' in experience
        assert experience['reward'] < 0  # Failed outcome should have negative reward
        assert experience['metadata']['success'] == False
        
        logger.info("✓ Convert to experience (failure) test passed")
    
    def test_reward_calculation(self, simple_problem, sample_plan):
        """Test reward calculation logic"""
        converter = ProblemToExperienceConverter()
        
        # Test with different metrics
        outcome1 = ExecutionOutcome(
            success=True,
            execution_time=30.0,
            metrics={'solution_quality': 0.9}
        )
        
        outcome2 = ExecutionOutcome(
            success=True,
            execution_time=30.0,
            metrics={'solution_quality': 0.5}
        )
        
        exp1 = converter.convert_to_experience(simple_problem, sample_plan, outcome1)
        exp2 = converter.convert_to_experience(simple_problem, sample_plan, outcome2)
        
        # Higher quality should yield higher reward
        assert exp1['reward'] > exp2['reward']
        
        logger.info("✓ Reward calculation test passed")
    
    def test_get_statistics(self, simple_problem, sample_plan, successful_outcome):
        """Test getting conversion statistics"""
        converter = ProblemToExperienceConverter()
        
        # Convert a few experiences
        for _ in range(3):
            converter.convert_to_experience(simple_problem, sample_plan, successful_outcome)
        
        stats = converter.get_statistics()
        
        assert 'total_conversions' in stats
        assert stats['total_conversions'] == 3
        assert 'failed_conversions' in stats
        
        logger.info("✓ Get statistics test passed")


# ============================================================
# DIFFICULTY ESTIMATOR TESTS
# ============================================================

class TestDecompositionDifficultyEstimator:
    """Test DecompositionDifficultyEstimator"""
    
    def test_estimator_initialization(self):
        """Test difficulty estimator initialization"""
        estimator = DecompositionDifficultyEstimator()
        
        assert len(estimator.difficulty_history) == 0
        
        logger.info("✓ Estimator initialization test passed")
    
    def test_estimate_simple_problem(self, simple_problem):
        """Test difficulty estimation for simple problem"""
        estimator = DecompositionDifficultyEstimator()
        
        difficulty = estimator.estimate(simple_problem)
        
        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0
        
        logger.info("✓ Simple problem difficulty estimation test passed")
    
    def test_estimate_complex_problem(self, complex_problem):
        """Test difficulty estimation for complex problem"""
        estimator = DecompositionDifficultyEstimator()
        
        difficulty = estimator.estimate(complex_problem)
        
        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0
        assert difficulty > 0.3  # Complex problems should be moderately difficult
        
        logger.info("✓ Complex problem difficulty estimation test passed")
    
    def test_difficulty_factors(self, simple_problem):
        """Test that difficulty factors are considered"""
        estimator = DecompositionDifficultyEstimator()
        
        # Add many constraints to increase difficulty
        simple_problem.metadata['constraints'] = [f'constraint_{i}' for i in range(10)]
        
        difficulty = estimator.estimate(simple_problem)
        
        # Should be higher than base difficulty due to constraints
        assert difficulty > 0.4  # Constraints increase but don't dominate
        
        logger.info("✓ Difficulty factors test passed")
    
    def test_difficulty_history(self, simple_problem):
        """Test difficulty history tracking"""
        estimator = DecompositionDifficultyEstimator()
        
        domain = simple_problem.metadata.get('domain', 'general')
        
        # Estimate multiple times
        for _ in range(5):
            estimator.estimate(simple_problem)
        
        # History should be tracked
        assert domain in estimator.difficulty_history
        assert len(estimator.difficulty_history[domain]) == 5
        
        logger.info("✓ Difficulty history test passed")


# ============================================================
# RLHF FEEDBACK ROUTER TESTS
# ============================================================

class TestRLHFFeedbackRouter:
    """Test RLHFFeedbackRouter"""
    
    def test_router_initialization(self):
        """Test router initialization"""
        router = RLHFFeedbackRouter()
        
        assert router.rlhf_manager is None
        assert router.feedback_count == 0
        
        logger.info("✓ Router initialization test passed")
    
    def test_route_outcome_to_feedback(self, simple_problem, sample_plan, 
                                      successful_outcome, mock_rlhf_manager):
        """Test routing outcome to RLHF feedback"""
        router = RLHFFeedbackRouter(mock_rlhf_manager)
        
        feedback = router.route_outcome_to_feedback(simple_problem, sample_plan, successful_outcome)
        
        assert feedback is not None
        assert mock_rlhf_manager.receive_feedback.called
        assert router.feedback_count == 1
        
        logger.info("✓ Route outcome to feedback test passed")
    
    def test_route_human_feedback(self, mock_rlhf_manager):
        """Test routing human feedback"""
        router = RLHFFeedbackRouter(mock_rlhf_manager)
        
        router.route_human_feedback('problem_sig_123', 0.8, 'Good solution')
        
        assert mock_rlhf_manager.receive_feedback.called
        
        logger.info("✓ Route human feedback test passed")
    
    def test_no_rlhf_manager(self, simple_problem, sample_plan, successful_outcome):
        """Test router without RLHF manager"""
        router = RLHFFeedbackRouter(None)
        
        feedback = router.route_outcome_to_feedback(simple_problem, sample_plan, successful_outcome)
        
        assert feedback is None
        
        logger.info("✓ No RLHF manager test passed")


# ============================================================
# INTEGRATED LEARNING COORDINATOR TESTS
# ============================================================

class TestIntegratedLearningCoordinator:
    """Test IntegratedLearningCoordinator"""
    
    def test_coordinator_initialization(self, mock_decomposer):
        """Test coordinator initialization"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer
        )
        
        assert coordinator.decomposer == mock_decomposer
        assert coordinator.continual_learner is None
        assert coordinator.curriculum_learner is None
        
        logger.info("✓ Coordinator initialization test passed")
    
    def test_coordinator_with_all_systems(self, mock_decomposer, mock_continual_learner,
                                         mock_curriculum_learner, mock_meta_learner,
                                         mock_metacognition, mock_rlhf_manager,
                                         mock_principle_learner):
        """Test coordinator with all learning systems"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=mock_continual_learner,
            curriculum_learner=mock_curriculum_learner,
            meta_learner=mock_meta_learner,
            metacognition=mock_metacognition,
            rlhf_manager=mock_rlhf_manager,
            principle_learner=mock_principle_learner
        )
        
        assert coordinator.continual_learner is not None
        assert coordinator.curriculum_learner is not None
        assert coordinator.meta_learner is not None
        assert coordinator.metacognition is not None
        assert coordinator.rlhf_manager is not None
        assert coordinator.principle_learner is not None
        
        logger.info("✓ Coordinator with all systems test passed")
    
    def test_learn_integrated(self, mock_decomposer, mock_continual_learner,
                             mock_principle_learner, simple_problem, 
                             sample_plan, successful_outcome):
        """Test integrated learning"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=mock_continual_learner,
            principle_learner=mock_principle_learner
        )
        
        coordinator.learn_integrated(simple_problem, sample_plan, successful_outcome)
        
        # Check that learning was called
        assert mock_continual_learner.process_experience.called
        assert mock_decomposer.learn_from_execution.called
        assert mock_principle_learner.extract_and_promote.called
        
        # Check statistics updated
        assert coordinator.integration_stats['total_learning_calls'] == 1
        assert coordinator.integration_stats['continual_learning_updates'] == 1
        assert coordinator.integration_stats['principles_extracted'] == 2
        
        logger.info("✓ Learn integrated test passed")
    
    def test_get_next_problems_curriculum(self, mock_decomposer, mock_curriculum_learner,
                                         simple_problem):
        """Test getting next problems from curriculum"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            curriculum_learner=mock_curriculum_learner
        )
        
        # Add problem to queue
        coordinator.problem_queue.append({
            'problem': simple_problem,
            'difficulty': 0.5,
            'timestamp': time.time()
        })
        
        # Get next batch
        batch = coordinator.get_next_problems_curriculum(batch_size=5)
        
        assert isinstance(batch, list)
        
        logger.info("✓ Get next problems curriculum test passed")
    
    def test_analyze_learning_effectiveness(self, mock_decomposer, mock_continual_learner):
        """Test analyzing learning effectiveness"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=mock_continual_learner
        )
        
        analysis = coordinator.analyze_learning_effectiveness()
        
        assert 'integration_stats' in analysis
        assert 'decomposer_stats' in analysis
        assert 'conversion_stats' in analysis
        assert 'continual_learning' in analysis
        
        logger.info("✓ Analyze learning effectiveness test passed")
    
    def test_get_recommendations(self, mock_decomposer, mock_continual_learner,
                                mock_metacognition):
        """Test getting learning recommendations"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=mock_continual_learner,
            metacognition=mock_metacognition
        )
        
        recommendations = coordinator.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        logger.info("✓ Get recommendations test passed")


# ============================================================
# UNIFIED DECOMPOSER LEARNER TESTS
# ============================================================

class TestUnifiedDecomposerLearner:
    """Test UnifiedDecomposerLearner"""
    
    def test_unified_learner_initialization(self):
        """Test unified learner initialization"""
        learner = UnifiedDecomposerLearner(
            enable_continual=False,
            enable_curriculum=False,
            enable_meta=False,
            enable_metacognition=False,
            enable_rlhf=False,
            enable_principle_learning=False
        )
        
        assert learner.decomposer is not None
        assert learner.coordinator is not None
        
        logger.info("✓ Unified learner initialization test passed")
    
    def test_create_unified_decomposer(self):
        """Test factory function"""
        learner = create_unified_decomposer(
            enable_all=False,
            enable_principle_learning=False
        )
        
        assert learner is not None
        assert isinstance(learner, UnifiedDecomposerLearner)
        
        logger.info("✓ Create unified decomposer test passed")


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestLearningIntegration:
    """Integration tests for learning system"""
    
    def test_full_learning_cycle_mock(self, simple_problem, sample_plan, 
                                     successful_outcome, mock_decomposer,
                                     mock_continual_learner, mock_principle_learner):
        """Test full learning cycle with mocks"""
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=mock_continual_learner,
            principle_learner=mock_principle_learner
        )
        
        # Run learning
        coordinator.learn_integrated(simple_problem, sample_plan, successful_outcome)
        
        # Verify all components called
        assert mock_decomposer.learn_from_execution.called
        assert mock_continual_learner.process_experience.called
        assert mock_principle_learner.extract_and_promote.called
        
        # Verify statistics
        stats = coordinator.integration_stats
        assert stats['total_learning_calls'] == 1
        assert stats['continual_learning_updates'] == 1
        assert stats['principles_extracted'] == 2
        assert stats['principles_promoted'] == 1
        
        logger.info("✓ Full learning cycle test passed")
    
    def test_error_handling(self, simple_problem, sample_plan, 
                           successful_outcome, mock_decomposer):
        """Test error handling in learning"""
        # Create coordinator with faulty learner
        faulty_learner = Mock()
        faulty_learner.process_experience = Mock(side_effect=Exception("Test error"))
        
        coordinator = IntegratedLearningCoordinator(
            decomposer=mock_decomposer,
            continual_learner=faulty_learner
        )
        
        # Should handle error gracefully
        coordinator.learn_integrated(simple_problem, sample_plan, successful_outcome)
        
        # Should still update statistics
        assert coordinator.integration_stats['total_learning_calls'] == 1
        assert coordinator.integration_stats['learning_errors'] == 1
        
        logger.info("✓ Error handling test passed")


# ============================================================
# PERFORMANCE TESTS
# ============================================================

class TestPerformance:
    """Performance tests"""
    
    def test_conversion_performance(self, simple_problem, sample_plan, successful_outcome):
        """Test conversion performance"""
        converter = ProblemToExperienceConverter()
        
        start_time = time.time()
        
        # Convert many experiences
        for _ in range(100):
            converter.convert_to_experience(simple_problem, sample_plan, successful_outcome)
        
        elapsed = time.time() - start_time
        
        # Should be fast
        assert elapsed < 1.0  # 100 conversions in under 1 second
        
        logger.info("✓ Conversion performance test passed (%.3f seconds)", elapsed)
    
    def test_difficulty_estimation_performance(self, complex_problem):
        """Test difficulty estimation performance"""
        estimator = DecompositionDifficultyEstimator()
        
        start_time = time.time()
        
        # Estimate many times
        for _ in range(100):
            estimator.estimate(complex_problem)
        
        elapsed = time.time() - start_time
        
        # Should be fast
        assert elapsed < 0.5  # 100 estimations in under 0.5 seconds
        
        logger.info("✓ Difficulty estimation performance test passed (%.3f seconds)", elapsed)


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])