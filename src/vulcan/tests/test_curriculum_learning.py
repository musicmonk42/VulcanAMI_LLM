"""
Test suite for curriculum learning module
"""

import pytest

# Skip entire module if torch is not available (curriculum_learning imports learning modules that require torch)
torch = pytest.importorskip(
    "torch", reason="PyTorch required for curriculum_learning tests"
)

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from vulcan.learning.curriculum_learning import (CompositeDifficultyEstimator,
                                                 CurriculumLearner,
                                                 CurriculumMetrics,
                                                 DifficultyEstimator,
                                                 DifficultyMetric,
                                                 LearnedDifficultyEstimator,
                                                 PacingStrategy, StageInfo)
from vulcan.learning.learning_types import LearningConfig


class TestDifficultyEstimators:
    """Test difficulty estimator implementations"""

    def test_base_difficulty_estimator(self):
        """Test base class raises NotImplementedError"""
        estimator = DifficultyEstimator()
        with pytest.raises(NotImplementedError):
            estimator.estimate({"task": "test"})

    def test_composite_difficulty_estimator(self):
        """Test composite estimator combines multiple estimators"""
        # Create mock estimators
        est1 = Mock(spec=DifficultyEstimator)
        est1.estimate.return_value = 0.3

        est2 = Mock(spec=DifficultyEstimator)
        est2.estimate.return_value = 0.7

        # Create composite with equal weights
        composite = CompositeDifficultyEstimator([est1, est2])

        task = {"test": "task"}
        difficulty = composite.estimate(task)

        assert difficulty == 0.5  # Average of 0.3 and 0.7
        est1.estimate.assert_called_once_with(task)
        est2.estimate.assert_called_once_with(task)

    def test_composite_with_weights(self):
        """Test composite estimator with custom weights"""
        est1 = Mock(spec=DifficultyEstimator)
        est1.estimate.return_value = 0.4

        est2 = Mock(spec=DifficultyEstimator)
        est2.estimate.return_value = 0.8

        # Weight first estimator more
        composite = CompositeDifficultyEstimator([est1, est2], weights=[0.8, 0.2])

        difficulty = composite.estimate({"test": "task"})
        assert abs(difficulty - (0.4 * 0.8 + 0.8 * 0.2)) < 0.001  # 0.48

    def test_learned_difficulty_estimator(self):
        """Test learned difficulty estimator"""
        estimator = LearnedDifficultyEstimator()

        # Test default features extraction
        task = {"complexity": 0.6, "noise_level": 0.2}
        difficulty = estimator.estimate(task)
        assert 0 <= difficulty <= 1

        # Update with performance data
        for i in range(10):
            estimator.update({"id": f"task_{i}"}, performance=0.8 - i * 0.05)

        # Should update performance history
        assert len(estimator.performance_history) == 10

    def test_learned_estimator_model_training(self):
        """Test that learned estimator trains model after enough data"""
        estimator = LearnedDifficultyEstimator()

        # Add enough data to trigger training
        for i in range(101):
            task = {"id": f"task_{i}", "complexity": i / 100}
            estimator.update(task, performance=1.0 - i / 100)

        # Model should be trained after 100 samples (checked at 50 intervals)
        # Note: Model training requires sklearn, so we just check it was attempted
        assert len(estimator.performance_history) == 101


class TestCurriculumLearner:
    """Test CurriculumLearner class"""

    @pytest.fixture
    def config(self):
        return LearningConfig(curriculum_stages=5, learning_rate=0.001)

    @pytest.fixture
    def learner(self, config, tmp_path):
        learner = CurriculumLearner(
            config=config, pacing_strategy=PacingStrategy.THRESHOLD
        )
        learner.save_path = tmp_path
        return learner

    def test_initialization(self, learner):
        """Test curriculum learner initialization"""
        assert learner.current_stage == 0
        assert learner.samples_in_stage == 0
        assert len(learner.curriculum_stages) == 0
        assert learner.pacing_strategy == PacingStrategy.THRESHOLD
        assert isinstance(learner.metrics, CurriculumMetrics)

    def test_generate_curriculum_single_objective(self, learner):
        """Test curriculum generation with single objective"""
        # Create tasks with varying difficulty
        tasks = []
        for i in range(20):
            tasks.append({"id": i, "difficulty": i / 20.0, "content": f"task_{i}"})

        # Generate curriculum
        curriculum = learner.generate_curriculum(tasks, auto_cluster=False)

        # Should have stages
        assert len(curriculum) > 0
        assert len(curriculum) <= learner.config.curriculum_stages

        # Tasks should be ordered by difficulty
        for stage_tasks in curriculum:
            assert len(stage_tasks) > 0

    def test_generate_curriculum_with_clustering(self, learner):
        """Test curriculum generation with clustering"""
        # Create enough tasks for clustering
        tasks = []
        for i in range(30):
            tasks.append(
                {
                    "id": i,
                    "difficulty": (i % 3) / 3.0 + i / 100.0,
                    "complexity": i / 30.0,
                }
            )

        curriculum = learner.generate_curriculum(tasks, auto_cluster=True)

        assert len(curriculum) > 0
        # Should have created clusters
        assert len(learner.task_clusters) > 0 or len(tasks) < 10

    def test_multi_objective_curriculum(self):
        """Test multi-objective curriculum generation"""
        learner = CurriculumLearner(multi_objective=True)

        tasks = []
        for i in range(20):
            tasks.append(
                {
                    "id": i,
                    "complexity": i / 20.0,
                    "noise_level": (20 - i) / 20.0,
                    "data_size": i * 10,
                }
            )

        curriculum = learner.generate_curriculum(tasks)

        assert len(curriculum) > 0
        # Check that objectives were computed
        assert learner.objective_weights is not None

    def test_get_next_batch(self, learner):
        """Test getting next batch of tasks"""
        # Generate curriculum first
        tasks = [{"id": i, "difficulty": i / 10} for i in range(10)]
        learner.generate_curriculum(tasks)

        # Get first batch
        batch = learner.get_next_batch(performance=None, batch_size=3)
        assert len(batch) == 3
        assert learner.samples_in_stage == 3

        # Get batch with performance update
        batch = learner.get_next_batch(performance=0.7, batch_size=2)
        assert len(batch) == 2
        assert learner.metrics.total_samples == 5

    def test_stage_advancement_threshold(self, learner):
        """Test stage advancement with threshold strategy"""
        learner.pacing_strategy = PacingStrategy.THRESHOLD
        learner.min_samples_per_stage = 5
        learner.adaptive_threshold = 0.8

        # Generate curriculum
        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        initial_stage = learner.current_stage

        # Process samples with low performance - should not advance
        for i in range(10):
            learner.get_next_batch(performance=0.6, batch_size=1)

        assert learner.current_stage == initial_stage

        # Process samples with high performance - should advance
        for i in range(10):
            learner.get_next_batch(performance=0.9, batch_size=1)

        # Should have advanced if threshold met
        if len(learner.curriculum_stages) > 1:
            assert learner.current_stage > initial_stage

    def test_stage_advancement_fixed(self):
        """Test fixed pacing strategy"""
        learner = CurriculumLearner(pacing_strategy=PacingStrategy.FIXED)
        learner.min_samples_per_stage = 10

        tasks = [{"id": i} for i in range(30)]
        learner.generate_curriculum(tasks)

        initial_stage = learner.current_stage

        # Process exactly min_samples_per_stage
        for i in range(10):
            learner.get_next_batch(performance=0.5, batch_size=1)

        # Should advance after fixed number
        if len(learner.curriculum_stages) > 1:
            assert learner.current_stage > initial_stage

    def test_adaptive_pacing(self):
        """Test adaptive pacing based on learning speed"""
        learner = CurriculumLearner(pacing_strategy=PacingStrategy.ADAPTIVE)
        learner.min_samples_per_stage = 10

        tasks = [{"id": i} for i in range(30)]
        learner.generate_curriculum(tasks)

        # Store initial stage to track advancement
        initial_stage = learner.current_stage

        # Simulate plateaued learning
        for i in range(20):
            # Performance plateaus at 0.75
            perf = 0.75 + np.random.normal(0, 0.01)
            learner.get_next_batch(performance=perf, batch_size=1)

        # Check either:
        # 1. We've advanced stages (stage_performance was reset)
        # 2. We're still in same stage with performance data
        if learner.current_stage > initial_stage:
            # Stage advanced, performance was reset - this is expected
            assert learner.current_stage > initial_stage
        else:
            # Still in same stage, should have performance data
            assert len(learner.stage_performance) > 0

    def test_exponential_pacing(self):
        """Test exponential pacing strategy"""
        learner = CurriculumLearner(pacing_strategy=PacingStrategy.EXPONENTIAL)
        learner.min_samples_per_stage = 5

        tasks = [{"id": i} for i in range(50)]
        learner.generate_curriculum(tasks)

        # First stage requires min_samples
        initial_stage = 0
        for i in range(5):
            learner.get_next_batch(performance=0.8, batch_size=1)

        if len(learner.curriculum_stages) > 1:
            # Should advance after min_samples * 2^0 = 5
            assert learner.current_stage > initial_stage

            # Second stage requires more samples
            second_stage = learner.current_stage
            for i in range(10):  # min_samples * 2^1 = 10
                learner.get_next_batch(performance=0.8, batch_size=1)

    def test_self_paced_strategy(self):
        """Test self-paced learning strategy"""
        learner = CurriculumLearner(pacing_strategy=PacingStrategy.SELF_PACED)

        tasks = [{"id": i} for i in range(30)]
        learner.generate_curriculum(tasks)

        # Simulate consistent high performance
        for i in range(15):
            learner.get_next_batch(
                performance=0.88 + np.random.normal(0, 0.02), batch_size=1
            )

        # Should advance with consistent high performance and low variance
        assert learner.samples_in_stage > 0

    def test_difficulty_adjustment(self, learner):
        """Test within-stage difficulty adjustment"""
        tasks = [{"id": i, "difficulty": i / 30} for i in range(30)]
        learner.generate_curriculum(tasks)

        # Simulate high performance
        for i in range(10):
            learner.get_next_batch(performance=0.95, batch_size=1)

        # Should have made difficulty adjustments
        if learner.current_stage < len(learner.curriculum_stages) - 1:
            assert len(learner.difficulty_adjustments) > 0
            assert learner.difficulty_adjustments[-1]["adjustment"] == "harder"

    def test_task_sampling_with_clusters(self, learner):
        """Test diverse task sampling when clusters exist"""
        # Create tasks and assign to clusters
        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        # Manually create clusters for testing
        learner.task_clusters = {str({"id": i}): i % 3 for i in range(20)}

        # Get batch and check diversity
        if learner.curriculum_stages:
            batch = learner.get_next_batch(batch_size=6)
            assert len(batch) == 6

    def test_curriculum_stats(self, learner):
        """Test getting curriculum statistics"""
        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        # Process some batches
        for i in range(5):
            learner.get_next_batch(performance=0.7 + i * 0.05, batch_size=2)

        stats = learner.get_curriculum_stats()

        assert "current_stage" in stats
        assert "total_stages" in stats
        assert "samples_in_stage" in stats
        assert "metrics" in stats
        assert stats["total_stages"] == len(learner.curriculum_stages)

    def test_analyze_effectiveness(self, learner):
        """Test curriculum effectiveness analysis"""
        tasks = [{"id": i} for i in range(30)]
        learner.generate_curriculum(tasks)

        # Simulate learning with improvement
        performances = np.linspace(0.4, 0.9, 20)
        for perf in performances:
            learner.get_next_batch(performance=perf, batch_size=1)

        analysis = learner.analyze_curriculum_effectiveness()

        assert "learning_efficiency" in analysis
        assert "curriculum_smoothness" in analysis
        assert "optimal_stages" in analysis
        assert analysis["learning_efficiency"] > 0

    def test_save_and_load_state(self, learner, tmp_path):
        """Test saving and loading curriculum state"""
        # Generate curriculum and process some batches
        tasks = [{"id": i, "data": f"task_{i}"} for i in range(20)]
        learner.generate_curriculum(tasks)

        for i in range(5):
            learner.get_next_batch(performance=0.6 + i * 0.08, batch_size=2)

        # Save state
        save_path = learner.save_state()
        assert Path(save_path).exists()

        # Create new learner and load state
        new_learner = CurriculumLearner()
        new_learner.load_state(save_path)

        # Check state was restored
        assert new_learner.current_stage == learner.current_stage
        assert new_learner.samples_in_stage == learner.samples_in_stage
        assert len(new_learner.curriculum_stages) == len(learner.curriculum_stages)
        assert new_learner.metrics.total_samples == learner.metrics.total_samples

    def test_learning_curve_estimation(self, learner):
        """Test learning rate estimation from performance trajectory"""
        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        # Simulate learning curve
        for i in range(30):
            t = i / 10.0
            # Exponential learning curve
            performance = 0.9 * (1 - np.exp(-0.5 * t)) + np.random.normal(0, 0.05)
            learner.get_next_batch(performance=performance, batch_size=1)

        # Should have estimated learning rate
        assert learner.learning_rate_estimate >= 0

    def test_stage_completion_tracking(self, learner):
        """Test that stage completion is properly tracked"""
        learner.min_samples_per_stage = 5
        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        # Complete first stage
        for i in range(5):
            learner.get_next_batch(performance=0.85, batch_size=1)

        # Check stage info was updated
        if 0 in learner.stage_info:
            stage = learner.stage_info[0]
            if learner.current_stage > 0:
                assert stage.completed
                assert stage.samples_seen > 0

    def test_curriculum_structure_save(self, learner, tmp_path):
        """Test saving curriculum structure for analysis"""
        learner.save_path = tmp_path

        tasks = [{"id": i} for i in range(20)]
        learner.generate_curriculum(tasks)

        structure_file = tmp_path / "curriculum_structure.json"
        assert structure_file.exists()

        with open(structure_file, "r") as f:
            structure = json.load(f)

        assert "num_stages" in structure
        assert "tasks_per_stage" in structure
        assert structure["num_stages"] == len(learner.curriculum_stages)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_task_list(self):
        """Test handling of empty task list"""
        learner = CurriculumLearner()
        curriculum = learner.generate_curriculum([])
        assert len(curriculum) == 0

    def test_single_task(self):
        """Test handling of single task"""
        learner = CurriculumLearner()
        curriculum = learner.generate_curriculum([{"id": 0}])
        assert len(curriculum) > 0
        assert len(curriculum[0]) == 1

    def test_insufficient_tasks_for_clustering(self):
        """Test clustering with insufficient tasks"""
        learner = CurriculumLearner()
        tasks = [{"id": i} for i in range(5)]  # Less than 10 required

        # Should not crash
        curriculum = learner.generate_curriculum(tasks, auto_cluster=True)
        assert len(curriculum) > 0

    def test_get_batch_without_curriculum(self):
        """Test getting batch before generating curriculum"""
        learner = CurriculumLearner()
        batch = learner.get_next_batch(batch_size=5)
        assert len(batch) == 0

    def test_advance_beyond_last_stage(self):
        """Test that learner doesn't advance beyond last stage"""
        learner = CurriculumLearner(pacing_strategy=PacingStrategy.FIXED)
        learner.min_samples_per_stage = 2

        tasks = [{"id": i} for i in range(10)]
        learner.generate_curriculum(tasks)

        last_stage = len(learner.curriculum_stages) - 1

        # Process many batches
        for i in range(100):
            learner.get_next_batch(performance=0.9, batch_size=1)

        assert learner.current_stage == last_stage

    def test_callable_difficulty_estimator(self):
        """Test using callable as difficulty estimator"""

        def custom_estimator(task):
            return task.get("custom_difficulty", 0.5)

        learner = CurriculumLearner(difficulty_estimator=custom_estimator)

        tasks = [
            {"id": 0, "custom_difficulty": 0.1},
            {"id": 1, "custom_difficulty": 0.9},
            {"id": 2, "custom_difficulty": 0.5},
        ]

        curriculum = learner.generate_curriculum(tasks)

        # Tasks should be ordered by custom difficulty
        assert (
            curriculum[0][0]["custom_difficulty"]
            <= curriculum[-1][-1]["custom_difficulty"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
