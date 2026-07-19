import asyncio
import io
import math
from collections import deque
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for learning containment tests")
import torch.nn as nn
import torch.optim as optim

from src.vulcan.api_gateway import APIGateway
from src.vulcan.learning import UnifiedLearningSystem
from src.vulcan.learning.continual_learning import EnhancedContinualLearner
from src.vulcan.learning.metacognition import (
    MetaCognitiveMonitor,
    MetacognitiveMutationRejected,
    RecommendationStatus,
)


class ExplodingRequest:
    async def json(self):
        raise AssertionError("request body must not be parsed")


@pytest.mark.asyncio
async def test_learn_handler_fails_closed_before_body_or_learner_resolution():
    gateway = APIGateway.__new__(APIGateway)
    gateway.deployment = Mock()
    response = await APIGateway.learn(gateway, ExplodingRequest())
    assert response.status == 501
    assert response.headers["Cache-Control"] == "no-store"
    body = response.text
    assert "learning_not_implemented" in body
    assert "Online learning is disabled until verification gates pass." in body
    assert not gateway.deployment.mock_calls


@pytest.mark.asyncio
async def test_learn_repeated_malformed_calls_are_side_effect_free():
    gateway = APIGateway.__new__(APIGateway)
    gateway.deployment = Mock()
    responses = [await APIGateway.learn(gateway, ExplodingRequest()) for _ in range(3)]
    assert [r.status for r in responses] == [501, 501, 501]
    assert not gateway.deployment.mock_calls


def test_unified_learning_system_defaults_progressive_disabled():
    system = UnifiedLearningSystem(enable_world_model=False, enable_curriculum=False, enable_metacognition=False)
    learner = system.continual_learner
    assert learner is not None
    assert learner.use_progressive is False
    assert not hasattr(learner, "progressive_network")
    assert all("progressive" not in k for k in learner.optimizers)


def test_unified_learning_system_rejects_progressive_activation():
    with pytest.raises(RuntimeError, match="Progressive learning is disabled"):
        UnifiedLearningSystem(enable_world_model=False, enable_curriculum=False, enable_metacognition=False, enable_progressive=True)


def test_enhanced_continual_learner_rejects_unverified_progressive_activation():
    with pytest.raises(RuntimeError, match="Progressive learning is disabled"):
        EnhancedContinualLearner(use_progressive=True)


def test_saved_state_cannot_reactivate_progressive(tmp_path):
    learner = EnhancedContinualLearner(use_progressive=False, use_hierarchical=False)
    path = tmp_path / "state.pkl"
    saved = learner.save_state(path)
    import pickle
    with open(saved, "rb") as f:
        state = pickle.load(f)
    state["use_progressive"] = True
    state["model_state"]["progressive_network.columns.0.0.weight"] = torch.zeros(1)
    with open(saved, "wb") as f:
        pickle.dump(state, f)

    reopened = EnhancedContinualLearner(use_progressive=False, use_hierarchical=False)
    reopened.load_state(saved)
    assert reopened.use_progressive is False
    assert not hasattr(reopened, "progressive_network")


def _model_optimizer():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2), nn.Linear(2, 1))
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    return model, optimizer


def _state_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def test_metacognition_observe_only_recommendations_do_not_mutate():
    model, optimizer = _model_optimizer()
    monitor = MetaCognitiveMonitor(model, optimizer)
    before = _state_bytes(model)
    opt_id = id(optimizer)
    params = [id(p) for group in optimizer.param_groups for p in group["params"]]
    lr = optimizer.param_groups[0]["lr"]
    wd = optimizer.param_groups[0]["weight_decay"]
    dropouts = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]

    for _ in range(25):
        monitor.update_self_model({"loss": 0.9, "modality": "text", "predicted_confidence": 0.9, "actual_performance": 0.1})
    analysis = monitor.analyze_learning_efficiency()

    assert _state_bytes(model) == before
    assert id(optimizer) == opt_id
    assert [id(p) for group in optimizer.param_groups for p in group["params"]] == params
    assert optimizer.param_groups[0]["lr"] == lr
    assert optimizer.param_groups[0]["weight_decay"] == wd
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == dropouts
    assert analysis["recommendations"]
    assert all(r["status"] == RecommendationStatus.NOT_APPLIED.value for r in analysis["recommendations"])
    assert list(monitor.applied_improvements) == []


def test_metacognition_direct_mutation_boundary_rejected():
    monitor = MetaCognitiveMonitor(*_model_optimizer())
    with pytest.raises(MetacognitiveMutationRejected):
        monitor._apply_improvements({"recommendations": [{"issue": "high_loss", "auto_fix": True}]})
    assert list(monitor.applied_improvements) == []


def test_metacognition_non_finite_telemetry_rejected():
    monitor = MetaCognitiveMonitor(*_model_optimizer())
    with pytest.raises(ValueError):
        monitor.update_self_model({"loss": math.inf, "modality": "text"})
