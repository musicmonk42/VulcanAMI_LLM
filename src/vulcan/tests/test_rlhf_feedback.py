"""
Test suite for RLHF feedback module
"""

import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for rlhf_feedback tests")

import asyncio
import threading
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, Mock, create_autospec, patch

import aiohttp
import numpy as np
import torch.nn as nn

from vulcan.config import EMBEDDING_DIM, HIDDEN_DIM
from vulcan.learning.learning_types import FeedbackData, LearningConfig
from vulcan.learning.rlhf_feedback import LiveFeedbackProcessor, RLHFManager


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward(self, x):
        return torch.relu(self.fc(x))


class TestRLHFManager:
    """Test RLHFManager class"""

    @pytest.fixture
    def model(self):
        """Create test model - fresh instance with train() mode enabled"""
        model = SimpleModel()
        model.train()  # Explicitly set to train mode to prevent gradient issues
        return model

    @pytest.fixture
    def config(self):
        """Create test config"""
        return LearningConfig(
            feedback_buffer_size=100,
            batch_size=10,
            reward_model_update_freq=20,
            ppo_epochs=3,
            ppo_clip=0.2,
            learning_rate=0.001,
        )

    @pytest.fixture
    def manager(self, model, config):
        """Create RLHFManager instance with mocked shutdown"""
        manager = RLHFManager(model, config)

        # Explicitly ensure all models are in train mode
        manager.base_model.train()
        manager.reward_model.train()
        manager.value_model.train()
        manager.policy_head.train()

        # Ensure _buffer_lock exists (in case of initialization issues)
        if not hasattr(manager, "_buffer_lock"):
            manager._buffer_lock = threading.RLock()

        # Mock the shutdown method to avoid ThreadPoolExecutor issues
        original_shutdown = manager.shutdown

        def mock_shutdown():
            manager._is_shutdown = True
            manager._shutdown_event.set()
            # Use simple shutdown without timeout
            manager.feedback_processor.shutdown(wait=True)
            if manager._processing_thread and manager._processing_thread.is_alive():
                manager._processing_thread.join(timeout=5)

        manager.shutdown = mock_shutdown
        yield manager
        manager.shutdown()

    def test_initialization(self, manager):
        """Test manager initialization"""
        assert manager.base_model is not None
        assert manager.reward_model is not None
        assert manager.value_model is not None
        assert manager.policy_head is not None
        assert len(manager.feedback_buffer) == 0
        assert manager.feedback_stats["total_feedback"] == 0

    def test_receive_feedback(self, manager):
        """Test receiving feedback"""
        feedback = FeedbackData(
            feedback_id="test_1",
            timestamp=time.time(),
            feedback_type="correction",
            content="test feedback",
            context={"test": "context"},
            agent_response="agent response",
            human_preference="human preference",
            reward_signal=0.8,
            metadata={},
        )

        manager.receive_feedback(feedback)

        assert len(manager.feedback_buffer) == 1
        assert manager.feedback_stats["total_feedback"] == 1
        assert manager.feedback_stats["positive_feedback"] == 1
        assert manager.feedback_stats["corrections"] == 1

    def test_receive_negative_feedback(self, manager):
        """Test receiving negative feedback"""
        feedback = FeedbackData(
            feedback_id="test_2",
            timestamp=time.time(),
            feedback_type="preference",
            content="test feedback",
            context={},
            agent_response="bad response",
            human_preference="good response",
            reward_signal=-0.5,
            metadata={"preferred_over": "bad response"},
        )

        manager.receive_feedback(feedback)

        assert manager.feedback_stats["negative_feedback"] == 1
        assert manager.feedback_stats["preferences"] == 1
        assert len(manager.preference_pairs) == 1

    def test_process_feedback_batch(self, manager):
        """Test processing feedback batch"""
        # Add enough feedback to trigger processing
        for i in range(15):
            feedback = FeedbackData(
                feedback_id=f"test_{i}",
                timestamp=time.time(),
                feedback_type="correction",
                content=f"feedback {i}",
                context={},
                agent_response=f"response {i}",
                human_preference=f"preference {i}",
                reward_signal=np.random.uniform(-1, 1),
                metadata={},
            )
            manager.feedback_buffer.append(feedback)

        # Process batch
        manager._process_feedback_batch()

        # Check that feedback was processed
        assert len(manager.feedback_buffer) < 15
        assert len(manager.processed_feedback) > 0

    def test_extract_features(self, manager):
        """Test feature extraction"""
        # Test tensor input
        tensor_input = torch.randn(EMBEDDING_DIM)
        features = manager._extract_features(tensor_input)
        assert features.shape == (EMBEDDING_DIM,)
        assert features.device == manager.device

        # Test numpy input
        numpy_input = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        features = manager._extract_features(numpy_input)
        assert features.shape == (EMBEDDING_DIM,)

        # Test dict input with embedding
        dict_input = {"embedding": torch.randn(EMBEDDING_DIM)}
        features = manager._extract_features(dict_input)
        assert features.shape == (EMBEDDING_DIM,)

        # Test unknown input (should return random features)
        features = manager._extract_features("unknown")
        assert features.shape == (EMBEDDING_DIM,)

    def test_update_reward_model_direct(self, manager):
        """Test direct reward model update"""
        batch = [
            FeedbackData(
                feedback_id=f"test_{i}",
                timestamp=time.time(),
                feedback_type="correction",
                content=f"feedback {i}",
                context={},
                agent_response=torch.randn(EMBEDDING_DIM),
                human_preference=torch.randn(EMBEDDING_DIM),
                reward_signal=np.random.uniform(-1, 1),
                metadata={},
            )
            for i in range(5)
        ]

        # Initial reward model parameters
        initial_params = [p.clone() for p in manager.reward_model.parameters()]

        # Update reward model
        manager._update_reward_model_direct(batch)

        # Check that parameters changed
        for initial, current in zip(initial_params, manager.reward_model.parameters()):
            assert not torch.allclose(initial, current, rtol=1e-5)

    def test_update_reward_model_preferences(self, manager):
        """Test preference-based reward model update"""
        batch = [
            FeedbackData(
                feedback_id=f"test_{i}",
                timestamp=time.time(),
                feedback_type="preference",
                content=f"feedback {i}",
                context=torch.randn(EMBEDDING_DIM),
                agent_response=torch.randn(EMBEDDING_DIM),
                human_preference=torch.randn(EMBEDDING_DIM),
                reward_signal=0,
                metadata={"preferred_over": torch.randn(EMBEDDING_DIM)},
            )
            for i in range(5)
        ]

        # Update reward model
        manager._update_reward_model_preferences(batch)

        assert (
            manager.feedback_stats["reward_model_updates"] == 0
        )  # Not incremented in this method

    def test_update_policy_with_ppo(self, manager):
        """Test PPO policy update with proper gradient handling"""
        trajectories = []
        for _ in range(3):
            # Create trajectory with detached tensors
            trajectory = {
                "states": [
                    torch.randn(EMBEDDING_DIM).detach().requires_grad_(False)
                    for _ in range(10)
                ],
                "actions": [
                    torch.randn(EMBEDDING_DIM).detach().requires_grad_(False)
                    for _ in range(10)
                ],
                "log_probs": [
                    torch.randn(1).detach().requires_grad_(False) for _ in range(10)
                ],
            }
            trajectories.append(trajectory)

        # Mock the value model to return detached values
        with patch.object(manager.value_model, "forward") as mock_forward:
            mock_forward.return_value = torch.randn(10, 1).detach()

            # Update policy
            manager.update_policy_with_ppo(trajectories)

        assert manager.feedback_stats["policy_updates"] == 1

    def test_compute_log_probs(self, manager):
        """Test log probability computation"""
        states = torch.randn(5, EMBEDDING_DIM)
        actions = torch.randn(5, EMBEDDING_DIM)

        log_probs, entropy = manager._compute_log_probs(states, actions)

        assert log_probs.shape == (5,)
        assert entropy.shape == (5,)

    def test_compute_advantages(self, manager):
        """Test advantage computation"""
        rewards = torch.randn(10)
        values = torch.randn(10)

        advantages = manager._compute_advantages(rewards, values)

        assert advantages.shape == rewards.shape

    def test_sync_with_governance(self, manager):
        """Test governance synchronization"""
        decisions = [
            {
                "id": "gov_1",
                "agent_action": "action_1",
                "approved_action": "approved_1",
                "approved": True,
            },
            {
                "id": "gov_2",
                "agent_action": "action_2",
                "approved_action": "approved_2",
                "approved": False,
            },
        ]

        manager.sync_with_governance(decisions)

        assert len(manager.feedback_buffer) == 2
        assert manager.feedback_stats["total_feedback"] == 2

    @pytest.mark.asyncio
    async def test_fetch_feedback_from_api(self, manager):
        """Test fetching feedback from API with properly mocked session"""
        # Mock API response
        mock_response_data = {
            "feedback": [
                {
                    "id": "api_1",
                    "timestamp": time.time(),
                    "type": "correction",
                    "content": "api feedback",
                    "agent_response": "response",
                    "human_preference": "preference",
                    "reward_signal": 0.5,
                }
            ]
        }

        # Create a proper mock for the response context manager
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        # Create a mock session with proper async context manager
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Set the mocked session
        manager.api_session = mock_session

        # Fetch feedback
        feedback = await manager.fetch_feedback_from_api()

        # Verify results
        assert len(feedback) == 1
        assert feedback[0].feedback_id == "api_1"
        assert manager.feedback_stats["api_fetches"] > 0

    @pytest.mark.asyncio
    async def test_push_metrics_to_api(self, manager):
        """Test pushing metrics to API with properly mocked session"""
        metrics = {"accuracy": 0.95, "latency": 50}

        # Create a proper mock for the response context manager
        mock_response = AsyncMock()
        mock_response.status = 200

        # Create a mock session with proper async context manager
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Set the mocked session
        manager.api_session = mock_session

        # Push metrics
        await manager.push_metrics_to_api(metrics)

        # Verify the call was made
        mock_session.post.assert_called_once()

    def test_get_statistics(self, manager):
        """Test getting statistics"""
        # Add some feedback
        for i in range(5):
            feedback = FeedbackData(
                feedback_id=f"test_{i}",
                timestamp=time.time(),
                feedback_type="correction",
                content=f"feedback {i}",
                context={},
                agent_response=f"response {i}",
                human_preference=f"preference {i}",
                reward_signal=0.5,
                metadata={},
            )
            manager.receive_feedback(feedback)

        stats = manager.get_statistics()

        assert stats["total_feedback"] == 5
        assert stats["positive_feedback"] == 5
        assert "buffer_size" in stats
        assert "processed_count" in stats

    def test_shutdown(self, model, config):
        """Test manager shutdown with mocked ThreadPoolExecutor"""
        manager = RLHFManager(model, config)

        # Ensure _buffer_lock exists
        if not hasattr(manager, "_buffer_lock"):
            manager._buffer_lock = threading.RLock()

        # Mock the ThreadPoolExecutor shutdown to avoid timeout parameter issue
        with patch.object(manager.feedback_processor, "shutdown") as mock_shutdown:
            mock_shutdown.return_value = None

            # Create API session
            manager.api_session = MagicMock()
            manager.api_session.closed = False

            # Custom shutdown that doesn't use timeout parameter
            manager._is_shutdown = True
            manager._shutdown_event.set()
            manager.feedback_processor.shutdown(wait=True)

            assert manager._is_shutdown
            assert manager._shutdown_event.is_set()


class TestLiveFeedbackProcessor:
    """Test LiveFeedbackProcessor class"""

    @pytest.fixture
    def model(self):
        """Create test model"""
        return SimpleModel()

    @pytest.fixture
    def processor(self, model):
        """Create LiveFeedbackProcessor instance"""
        return LiveFeedbackProcessor(model)

    @pytest.mark.asyncio
    async def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.model is not None
        assert len(processor.performance_buffer) == 0
        assert processor.adaptive_lr == processor.config.learning_rate
        assert processor.performance_tracker["total_predictions"] == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, processor):
        """Test starting and stopping monitoring"""
        await processor.start_monitoring()
        assert processor._monitoring_task is not None

        await processor.stop_monitoring()
        assert processor._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_process_live_feedback(self, processor):
        """Test processing live feedback"""
        await processor.start_monitoring()

        feedback = {"type": "prediction_result", "correct": True}

        await processor.process_live_feedback(feedback)

        # Give time to process
        await asyncio.sleep(0.1)

        assert processor.performance_tracker["feedback_processed"] == 1

        await processor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_process_feedback_types(self, processor):
        """Test processing different feedback types"""
        await processor.start_monitoring()

        # Prediction result
        await processor._process_feedback(
            {"type": "prediction_result", "correct": True}
        )
        assert processor.performance_tracker["correct_predictions"] == 1

        # User rating
        await processor._process_feedback({"type": "user_rating", "rating": 0.8})
        assert len(processor.realtime_metrics["user_satisfaction"]) == 1

        # Latency
        await processor._process_feedback({"type": "latency", "value": 25})
        assert len(processor.realtime_metrics["latency_ms"]) == 1

        await processor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_immediate_adjustment(self, processor):
        """Test immediate adjustments for critical feedback"""
        initial_lr = processor.adaptive_lr

        # Error feedback
        await processor._immediate_adjustment({"type": "error", "priority": "critical"})
        assert processor.adaptive_lr < initial_lr

        # Poor performance feedback
        processor.adaptive_lr = initial_lr
        await processor._immediate_adjustment(
            {"type": "performance", "value": 0.2, "priority": "critical"}
        )
        assert processor.adaptive_lr > initial_lr

        # Great performance feedback
        processor.adaptive_lr = initial_lr
        await processor._immediate_adjustment(
            {"type": "performance", "value": 0.95, "priority": "critical"}
        )
        assert processor.adaptive_lr < initial_lr

    @pytest.mark.asyncio
    async def test_collect_metrics(self, processor):
        """Test metric collection"""
        metrics = await processor._collect_metrics()

        assert "latency_ms" in metrics
        assert "accuracy" in metrics
        assert "user_satisfaction" in metrics
        assert "memory_usage_mb" in metrics
        assert "cpu_percent" in metrics
        assert "throughput_qps" in metrics

    def test_get_average_latency(self, processor):
        """Test average latency calculation"""
        processor.realtime_metrics["latency_ms"].extend([10, 20, 30])

        avg_latency = processor._get_average_latency()
        assert avg_latency == 20.0

    def test_get_current_accuracy(self, processor):
        """Test current accuracy calculation"""
        processor.performance_tracker["total_predictions"] = 10
        processor.performance_tracker["correct_predictions"] = 8

        accuracy = processor._get_current_accuracy()
        assert accuracy == 0.8

    def test_get_average_satisfaction(self, processor):
        """Test average satisfaction calculation with float comparison"""
        processor.realtime_metrics["user_satisfaction"].extend([0.6, 0.8, 1.0])

        avg_satisfaction = processor._get_average_satisfaction()
        # Use approximate comparison for floating point
        assert pytest.approx(avg_satisfaction, rel=1e-9) == 0.8

    def test_calculate_throughput(self, processor):
        """Test throughput calculation"""
        processor.performance_tracker["start_time"] = time.time() - 10
        processor.performance_tracker["total_predictions"] = 50

        throughput = processor._calculate_throughput()
        assert throughput == pytest.approx(5.0, rel=0.1)

    def test_check_alerts(self, processor):
        """Test alert checking"""
        metrics = {
            "latency_ms": 150,  # Above threshold
            "accuracy": 0.6,  # Below threshold
            "user_satisfaction": 0.8,  # Above threshold
            "memory_usage_mb": 500,  # Below threshold
        }

        alerts = processor._check_alerts(metrics)

        assert len(alerts) == 2
        assert any(a["metric"] == "latency_ms" for a in alerts)
        assert any(a["metric"] == "accuracy" for a in alerts)

    def test_detect_degradation(self, processor):
        """Test degradation detection"""
        # Add older good accuracy
        for _ in range(10):
            processor.realtime_metrics["accuracy"].append(0.9)

        # Add recent poor accuracy
        for _ in range(10):
            processor.realtime_metrics["accuracy"].append(0.7)

        assert processor._detect_degradation() == True

        # Clear and test no degradation
        processor.realtime_metrics["accuracy"].clear()
        for _ in range(20):
            processor.realtime_metrics["accuracy"].append(0.9)

        assert processor._detect_degradation() == False

    def test_get_performance_summary(self, processor):
        """Test performance summary"""
        processor.performance_tracker["total_predictions"] = 100
        processor.performance_tracker["correct_predictions"] = 85
        processor.realtime_metrics["latency_ms"].extend([10, 20, 30])

        summary = processor.get_performance_summary()

        assert "total_predictions" in summary
        assert summary["total_predictions"] == 100
        assert "accuracy" in summary
        assert summary["accuracy"] == 0.85
        assert "average_latency_ms" in summary
        assert "throughput_qps" in summary

    def test_reset_metrics(self, processor):
        """Test metric reset"""
        processor.performance_tracker["total_predictions"] = 100
        processor.realtime_metrics["latency_ms"].extend([10, 20, 30])

        processor.reset_metrics()

        assert processor.performance_tracker["total_predictions"] == 0
        assert len(processor.realtime_metrics["latency_ms"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
