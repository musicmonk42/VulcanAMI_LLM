"""
Test suite for learning types module
"""

import pytest

# Skip entire module if torch is not available (learning_types module requires torch)
torch = pytest.importorskip("torch", reason="PyTorch required for learning_types tests")

import time
from dataclasses import asdict, fields
from unittest.mock import MagicMock, patch

import numpy as np

from vulcan.learning.learning_types import (FeedbackData, LearningConfig,
                                            LearningMode, LearningTrajectory,
                                            TaskInfo)


class TestLearningMode:
    """Test LearningMode enum"""

    def test_learning_modes_defined(self):
        """Test all learning modes are defined"""
        expected_modes = [
            "supervised",
            "unsupervised",
            "reinforcement",
            "meta",
            "continual",
            "curriculum",
            "transfer",
            "federated",
            "rlhf",
            "online",
        ]

        actual_modes = [mode.value for mode in LearningMode]
        assert set(expected_modes) == set(actual_modes)

    def test_learning_mode_access(self):
        """Test accessing learning modes"""
        assert LearningMode.SUPERVISED.value == "supervised"
        assert LearningMode.RLHF.value == "rlhf"
        assert LearningMode.META.value == "meta"

    def test_learning_mode_comparison(self):
        """Test learning mode comparisons"""
        mode1 = LearningMode.SUPERVISED
        mode2 = LearningMode.SUPERVISED
        mode3 = LearningMode.UNSUPERVISED

        assert mode1 == mode2
        assert mode1 != mode3
        assert mode1 is LearningMode.SUPERVISED


class TestLearningConfig:
    """Test LearningConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = LearningConfig()

        # Test core learning parameters
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.ewc_lambda == 100.0
        assert config.meta_lr == 0.001
        assert config.inner_lr == 0.01

        # Test buffer and threshold parameters
        assert config.replay_buffer_size == 10000
        assert config.consolidation_threshold == 100
        assert config.curriculum_stages == 5
        assert config.task_detection_threshold == 0.3

        # Test RLHF parameters
        assert config.rlhf_enabled == True
        assert config.feedback_buffer_size == 5000
        assert config.reward_model_update_freq == 100
        assert config.ppo_epochs == 4
        assert config.ppo_clip == 0.2
        assert config.kl_penalty == 0.01

        # Test checkpoint parameters
        assert config.checkpoint_frequency == 1000
        assert config.max_checkpoints == 100
        assert config.audit_trail_enabled == True

    def test_custom_values(self):
        """Test creating config with custom values"""
        config = LearningConfig(
            learning_rate=0.01, batch_size=64, rlhf_enabled=False, meta_batch_size=8
        )

        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.rlhf_enabled == False
        assert config.meta_batch_size == 8

        # Check other values remain default
        assert config.ewc_lambda == 100.0
        assert config.curriculum_stages == 5

    def test_config_modification(self):
        """Test modifying configuration after creation"""
        config = LearningConfig()

        config.learning_rate = 0.005
        config.batch_size = 128
        config.rlhf_enabled = False

        assert config.learning_rate == 0.005
        assert config.batch_size == 128
        assert config.rlhf_enabled == False

    def test_config_as_dict(self):
        """Test converting config to dictionary"""
        config = LearningConfig(learning_rate=0.002, batch_size=16)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["learning_rate"] == 0.002
        assert config_dict["batch_size"] == 16
        assert "rlhf_enabled" in config_dict
        assert "checkpoint_frequency" in config_dict

    def test_all_fields_accessible(self):
        """Test all fields are properly accessible"""
        config = LearningConfig()

        # Get all field names
        field_names = [f.name for f in fields(config)]

        # Check each field is accessible
        for field_name in field_names:
            assert hasattr(config, field_name)
            value = getattr(config, field_name)
            assert value is not None  # All fields should have defaults


class TestTaskInfo:
    """Test TaskInfo dataclass"""

    def test_task_creation_minimal(self):
        """Test creating task with minimal info"""
        task = TaskInfo(task_id="task_001", task_type="classification")

        assert task.task_id == "task_001"
        assert task.task_type == "classification"
        assert task.difficulty == 0.5  # Default
        assert task.samples_seen == 0
        assert task.performance == 0.0
        assert task.signature is None
        assert isinstance(task.metadata, dict)
        assert len(task.metadata) == 0

    def test_task_creation_full(self):
        """Test creating task with all parameters"""
        signature = np.random.randn(10)
        metadata = {"source": "dataset_a", "version": 2}

        task = TaskInfo(
            task_id="task_002",
            task_type="regression",
            difficulty=0.8,
            samples_seen=100,
            performance=0.75,
            signature=signature,
            metadata=metadata,
        )

        assert task.task_id == "task_002"
        assert task.task_type == "regression"
        assert task.difficulty == 0.8
        assert task.samples_seen == 100
        assert task.performance == 0.75
        assert np.array_equal(task.signature, signature)
        assert task.metadata == metadata

    def test_task_timestamp(self):
        """Test task creation timestamp"""
        # Create task and check it has a timestamp
        task = TaskInfo(task_id="task_003", task_type="clustering")

        # Timestamp should be set and recent
        assert task.created_at > 0
        assert isinstance(task.created_at, float)
        # Should be within last minute
        assert abs(time.time() - task.created_at) < 60

    def test_task_timestamp_custom(self):
        """Test task with custom creation timestamp"""
        custom_time = 1234567890.0
        task = TaskInfo(
            task_id="task_003b", task_type="clustering", created_at=custom_time
        )

        assert task.created_at == custom_time

    def test_task_modification(self):
        """Test modifying task info after creation"""
        task = TaskInfo(task_id="task_004", task_type="generation")

        # Modify fields
        task.samples_seen = 50
        task.performance = 0.85
        task.difficulty = 0.6
        task.metadata["status"] = "completed"

        assert task.samples_seen == 50
        assert task.performance == 0.85
        assert task.difficulty == 0.6
        assert task.metadata["status"] == "completed"

    def test_task_signature_update(self):
        """Test updating task signature"""
        task = TaskInfo(task_id="task_005", task_type="detection")

        assert task.signature is None

        # Add signature
        new_signature = np.ones(20)
        task.signature = new_signature

        assert task.signature is not None
        assert np.array_equal(task.signature, new_signature)

    def test_task_as_dict(self):
        """Test converting task to dictionary"""
        task = TaskInfo(task_id="task_006", task_type="segmentation", difficulty=0.7)

        task_dict = asdict(task)

        assert isinstance(task_dict, dict)
        assert task_dict["task_id"] == "task_006"
        assert task_dict["task_type"] == "segmentation"
        assert task_dict["difficulty"] == 0.7
        assert "created_at" in task_dict
        assert "metadata" in task_dict


class TestFeedbackData:
    """Test FeedbackData dataclass"""

    def test_feedback_creation(self):
        """Test creating feedback data"""
        feedback = FeedbackData(
            feedback_id="fb_001",
            timestamp=1234567890.0,
            feedback_type="rating",
            content={"rating": 4},
            context={"session": "sess_123"},
            agent_response="Generated text response",
            human_preference=4,
            reward_signal=0.8,
        )

        assert feedback.feedback_id == "fb_001"
        assert feedback.timestamp == 1234567890.0
        assert feedback.feedback_type == "rating"
        assert feedback.content == {"rating": 4}
        assert feedback.context == {"session": "sess_123"}
        assert feedback.agent_response == "Generated text response"
        assert feedback.human_preference == 4
        assert feedback.reward_signal == 0.8
        assert isinstance(feedback.metadata, dict)
        assert len(feedback.metadata) == 0

    def test_feedback_types(self):
        """Test different feedback types"""
        feedback_types = ["rating", "comparison", "correction", "preference"]

        for fb_type in feedback_types:
            feedback = FeedbackData(
                feedback_id=f"fb_{fb_type}",
                timestamp=time.time(),
                feedback_type=fb_type,
                content={},
                context={},
                agent_response=None,
                human_preference=None,
                reward_signal=0.0,
            )
            assert feedback.feedback_type == fb_type

    def test_feedback_with_metadata(self):
        """Test feedback with metadata"""
        metadata = {"user_id": "user_123", "session_length": 300, "quality_score": 0.9}

        feedback = FeedbackData(
            feedback_id="fb_002",
            timestamp=time.time(),
            feedback_type="comparison",
            content={"option_a": "text1", "option_b": "text2"},
            context={"task": "summarization"},
            agent_response=["text1", "text2"],
            human_preference="text1",
            reward_signal=1.0,
            metadata=metadata,
        )

        assert feedback.metadata == metadata
        assert feedback.metadata["user_id"] == "user_123"
        assert feedback.metadata["quality_score"] == 0.9

    def test_feedback_modification(self):
        """Test modifying feedback after creation"""
        feedback = FeedbackData(
            feedback_id="fb_003",
            timestamp=time.time(),
            feedback_type="correction",
            content="",
            context={},
            agent_response="",
            human_preference="",
            reward_signal=0.0,
        )

        # Modify fields
        feedback.content = "Corrected text"
        feedback.reward_signal = 0.5
        feedback.metadata["processed"] = True

        assert feedback.content == "Corrected text"
        assert feedback.reward_signal == 0.5
        assert feedback.metadata["processed"] == True

    def test_feedback_complex_content(self):
        """Test feedback with complex content types"""
        # Array content
        array_content = np.array([1, 2, 3, 4, 5])
        feedback1 = FeedbackData(
            feedback_id="fb_array",
            timestamp=time.time(),
            feedback_type="preference",
            content=array_content,
            context={},
            agent_response=None,
            human_preference=None,
            reward_signal=0.0,
        )
        assert np.array_equal(feedback1.content, array_content)

        # Nested dict content
        nested_content = {"level1": {"level2": {"data": [1, 2, 3]}}}
        feedback2 = FeedbackData(
            feedback_id="fb_nested",
            timestamp=time.time(),
            feedback_type="rating",
            content=nested_content,
            context={},
            agent_response=None,
            human_preference=None,
            reward_signal=0.0,
        )
        assert feedback2.content == nested_content


class TestLearningTrajectory:
    """Test LearningTrajectory dataclass"""

    def test_trajectory_creation(self):
        """Test creating learning trajectory"""
        states = [np.random.randn(10) for _ in range(5)]
        actions = ["action1", "action2", "action3", "action4", "action5"]
        rewards = [0.1, 0.3, 0.5, 0.7, 0.9]
        losses = [2.0, 1.8, 1.5, 1.2, 0.9]
        snapshots = ["path1.pkl", "path2.pkl", "path3.pkl"]

        trajectory = LearningTrajectory(
            trajectory_id="traj_001",
            start_time=1000.0,
            end_time=2000.0,
            task_id="task_001",
            agent_id="agent_001",
            states=states,
            actions=actions,
            rewards=rewards,
            losses=losses,
            parameter_snapshots=snapshots,
        )

        assert trajectory.trajectory_id == "traj_001"
        assert trajectory.start_time == 1000.0
        assert trajectory.end_time == 2000.0
        assert trajectory.task_id == "task_001"
        assert trajectory.agent_id == "agent_001"
        assert len(trajectory.states) == 5
        assert len(trajectory.actions) == 5
        assert trajectory.rewards == rewards
        assert trajectory.losses == losses
        assert trajectory.parameter_snapshots == snapshots
        assert isinstance(trajectory.metadata, dict)

    def test_trajectory_ongoing(self):
        """Test trajectory that is still ongoing (no end time)"""
        trajectory = LearningTrajectory(
            trajectory_id="traj_002",
            start_time=1000.0,
            end_time=None,
            task_id="task_002",
            agent_id="agent_002",
            states=[],
            actions=[],
            rewards=[],
            losses=[],
            parameter_snapshots=[],
        )

        assert trajectory.end_time is None
        assert len(trajectory.states) == 0
        assert len(trajectory.actions) == 0

    def test_trajectory_append_data(self):
        """Test appending data to trajectory"""
        trajectory = LearningTrajectory(
            trajectory_id="traj_003",
            start_time=1000.0,
            end_time=None,
            task_id="task_003",
            agent_id="agent_003",
            states=[],
            actions=[],
            rewards=[],
            losses=[],
            parameter_snapshots=[],
        )

        # Append data
        new_state = np.random.randn(10)
        trajectory.states.append(new_state)
        trajectory.actions.append("new_action")
        trajectory.rewards.append(0.5)
        trajectory.losses.append(1.5)
        trajectory.parameter_snapshots.append("snapshot.pkl")

        assert len(trajectory.states) == 1
        assert np.array_equal(trajectory.states[0], new_state)
        assert trajectory.actions[0] == "new_action"
        assert trajectory.rewards[0] == 0.5
        assert trajectory.losses[0] == 1.5
        assert trajectory.parameter_snapshots[0] == "snapshot.pkl"

    def test_trajectory_with_metadata(self):
        """Test trajectory with metadata"""
        metadata = {
            "experiment": "exp_001",
            "hyperparameters": {"lr": 0.001, "batch_size": 32},
            "environment": "production",
        }

        trajectory = LearningTrajectory(
            trajectory_id="traj_004",
            start_time=1000.0,
            end_time=2000.0,
            task_id="task_004",
            agent_id="agent_004",
            states=[],
            actions=[],
            rewards=[],
            losses=[],
            parameter_snapshots=[],
            metadata=metadata,
        )

        assert trajectory.metadata == metadata
        assert trajectory.metadata["experiment"] == "exp_001"
        assert trajectory.metadata["hyperparameters"]["lr"] == 0.001

    def test_trajectory_duration(self):
        """Test calculating trajectory duration"""
        trajectory = LearningTrajectory(
            trajectory_id="traj_005",
            start_time=1000.0,
            end_time=3500.0,
            task_id="task_005",
            agent_id="agent_005",
            states=[],
            actions=[],
            rewards=[],
            losses=[],
            parameter_snapshots=[],
        )

        duration = trajectory.end_time - trajectory.start_time
        assert duration == 2500.0

    def test_trajectory_statistics(self):
        """Test computing statistics from trajectory"""
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        losses = [2.0, 1.5, 1.0, 0.8, 0.6]

        trajectory = LearningTrajectory(
            trajectory_id="traj_006",
            start_time=1000.0,
            end_time=2000.0,
            task_id="task_006",
            agent_id="agent_006",
            states=[np.zeros(10) for _ in range(5)],
            actions=["a"] * 5,
            rewards=rewards,
            losses=losses,
            parameter_snapshots=[],
        )

        # Compute statistics
        avg_reward = sum(trajectory.rewards) / len(trajectory.rewards)
        avg_loss = sum(trajectory.losses) / len(trajectory.losses)

        assert avg_reward == 0.3
        # FIX: Use pytest.approx() for floating point comparison
        assert avg_loss == pytest.approx(1.18)
        assert max(trajectory.rewards) == 0.5
        assert min(trajectory.losses) == 0.6

    def test_trajectory_as_dict(self):
        """Test converting trajectory to dictionary"""
        trajectory = LearningTrajectory(
            trajectory_id="traj_007",
            start_time=1000.0,
            end_time=2000.0,
            task_id="task_007",
            agent_id="agent_007",
            states=[np.zeros(5)],
            actions=["action"],
            rewards=[0.5],
            losses=[1.0],
            parameter_snapshots=["snap.pkl"],
        )

        traj_dict = asdict(trajectory)

        assert isinstance(traj_dict, dict)
        assert traj_dict["trajectory_id"] == "traj_007"
        assert traj_dict["start_time"] == 1000.0
        assert traj_dict["end_time"] == 2000.0
        assert len(traj_dict["states"]) == 1
        assert len(traj_dict["actions"]) == 1


class TestIntegration:
    """Integration tests for learning types"""

    def test_complete_learning_scenario(self):
        """Test a complete learning scenario with all types"""
        # Create config
        config = LearningConfig(
            learning_rate=0.001, rlhf_enabled=True, checkpoint_frequency=100
        )

        # Create task
        task = TaskInfo(
            task_id="integration_task", task_type="multi_objective", difficulty=0.7
        )

        # Start trajectory
        start_time = time.time()
        trajectory = LearningTrajectory(
            trajectory_id="integration_traj",
            start_time=start_time,
            end_time=None,
            task_id=task.task_id,
            agent_id="test_agent",
            states=[],
            actions=[],
            rewards=[],
            losses=[],
            parameter_snapshots=[],
        )

        # Simulate learning steps
        for i in range(10):
            state = np.random.randn(config.batch_size)
            action = f"action_{i}"
            reward = np.random.random()
            loss = 2.0 - i * 0.1

            trajectory.states.append(state)
            trajectory.actions.append(action)
            trajectory.rewards.append(reward)
            trajectory.losses.append(loss)

            # Update task info
            task.samples_seen += config.batch_size
            task.performance = 1.0 / (1.0 + loss)

            # Create feedback
            if i % 3 == 0:
                feedback = FeedbackData(
                    feedback_id=f"feedback_{i}",
                    timestamp=time.time(),
                    feedback_type="rating",
                    content={"rating": int(reward * 5)},
                    context={"step": i},
                    agent_response=action,
                    human_preference=int(reward * 5),
                    reward_signal=reward,
                )
                feedback.metadata["task_id"] = task.task_id

            # Add small delay to ensure time difference
            time.sleep(0.001)

        # End trajectory (ensure end_time > start_time)
        trajectory.end_time = time.time()

        # Verify results
        assert task.samples_seen == config.batch_size * 10
        assert task.performance > 0.3
        assert len(trajectory.states) == 10
        assert len(trajectory.actions) == 10
        assert len(trajectory.rewards) == 10
        assert len(trajectory.losses) == 10
        assert trajectory.end_time > trajectory.start_time

    def test_type_compatibility(self):
        """Test that types work together properly"""
        # Create instances of each type
        mode = LearningMode.META
        config = LearningConfig()
        task = TaskInfo("test", "test_type")
        feedback = FeedbackData(
            "fb_test", time.time(), "rating", {}, {}, None, None, 0.0
        )
        trajectory = LearningTrajectory(
            "traj_test", time.time(), None, "task", "agent", [], [], [], [], []
        )

        # Test they can be used together
        combined = {
            "mode": mode,
            "config": config,
            "task": task,
            "feedback": feedback,
            "trajectory": trajectory,
        }

        assert combined["mode"] == LearningMode.META
        assert isinstance(combined["config"], LearningConfig)
        assert isinstance(combined["task"], TaskInfo)
        assert isinstance(combined["feedback"], FeedbackData)
        assert isinstance(combined["trajectory"], LearningTrajectory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
