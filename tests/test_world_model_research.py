import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.vulcan.learning import UnifiedLearningSystem
from src.vulcan.learning.world_model import DiscreteActionSpace, IsolatedWorldModel
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner


def action_space():
    return DiscreteActionSpace(actions=((-1.0,), (0.0,), (1.0,)))


def make_model(seed=3):
    return IsolatedWorldModel(1, action_space(), seed=seed, hidden_dim=16)


def transitions(values):
    states = []
    actions = []
    next_states = []
    rewards = []
    for x in values:
        for a in (-1.0, 0.0, 1.0):
            nx = x + a
            states.append([x])
            actions.append([a])
            next_states.append([nx])
            # Known reward: move toward +2, so action +1 is obviously best near zero.
            rewards.append([1.0 - abs(2.0 - nx)])
    return {
        "state": torch.tensor(states, dtype=torch.float32),
        "action": torch.tensor(actions, dtype=torch.float32),
        "next_state": torch.tensor(next_states, dtype=torch.float32),
        "reward": torch.tensor(rewards, dtype=torch.float32),
    }


def train(model, batch, steps=220):
    last = None
    for _ in range(steps):
        last = model.train_step(batch)
    return last


def test_heldout_transition_and_reward_error_decrease():
    model = make_model(4)
    train_batch = transitions([-2.0, -1.0, 0.0, 1.0])
    heldout = transitions([1.5, -1.5])
    transition_before = model.transition_error(heldout)
    reward_before = model.reward_error(heldout)
    train(model, train_batch)
    assert model.transition_error(heldout) < transition_before * 0.5
    assert model.reward_error(heldout) < reward_before * 0.5


def test_curiosity_error_decreases_and_intrinsic_reward_detached_bounded():
    model = make_model(5)
    batch = transitions([-1.0, 0.0, 1.0])
    heldout = transitions([0.5])
    before = model.transition_error({**heldout})
    train(model, batch)
    after = model.transition_error(heldout)
    assert after < before
    reward = model.intrinsic_curiosity_reward(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]))
    assert reward.requires_grad is False
    assert torch.isfinite(reward).all()
    assert 0.0 <= float(reward.item()) <= 1.0


def test_discrete_planner_beats_random_and_fixed_baselines_and_is_deterministic():
    a = make_model(6)
    b = make_model(6)
    train_batch = transitions([-1.0, 0.0, 1.0, 2.0])
    train(a, train_batch); train(b, train_batch)
    state = torch.tensor([0.0])
    action_a, info_a = a.plan_discrete(state, horizon=2)
    action_b, info_b = b.plan_discrete(state, horizon=2)
    assert torch.equal(action_a, torch.tensor([1.0]))
    assert torch.equal(action_a, action_b)
    assert info_a["discounted_return"] == pytest.approx(info_b["discounted_return"], abs=1e-8)
    fixed_zero_return = sum((a.discount ** i) * (1.0 - abs(2.0 - 0.0)) for i in range(2))
    random_baseline = sum((a.discount ** i) * ((1.0 - abs(2.0 - (-1.0))) + (1.0 - abs(2.0 - 0.0)) + (1.0 - abs(2.0 - 1.0))) / 3.0 for i in range(2))
    assert info_a["discounted_return"] > fixed_zero_return + 0.5
    assert info_a["discounted_return"] > random_baseline + 0.5


def test_illegal_empty_and_dimension_mismatch_fail_closed():
    model = make_model(7)
    with pytest.raises(ValueError):
        model.predict(torch.tensor([0.0]), torch.tensor([2.0]))
    with pytest.raises(ValueError):
        model.predict(torch.tensor([0.0, 1.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError):
        IsolatedWorldModel(1, DiscreteActionSpace(actions=()), seed=1)


def test_save_reopen_identical_predictions_and_plan_and_corruption_fails(tmp_path):
    model = make_model(8)
    train(model, transitions([-1.0, 0.0, 1.0]))
    state = torch.tensor([0.0]); action = torch.tensor([1.0])
    pred = model.predict(state, action)
    plan = model.plan_discrete(state, horizon=2)
    path = tmp_path / "world.json"
    model.save_research_state(str(path))
    reopened = make_model(99)
    reopened.load_research_state(str(path))
    pred2 = reopened.predict(state, action)
    plan2 = reopened.plan_discrete(state, horizon=2)
    assert torch.allclose(pred[0], pred2[0], atol=0, rtol=0)
    assert torch.allclose(pred[1], pred2[1], atol=0, rtol=0)
    assert torch.equal(plan[0], plan2[0])
    doc = json.loads(path.read_text())
    doc["state_dim"] = 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        reopened.load_research_state(str(bad))


def test_production_world_model_disabled_and_no_active_learning_capability():
    system = UnifiedLearningSystem(enable_continual=False, enable_curriculum=False, enable_meta_learning=False, enable_rlhf=False)
    assert system.world_model is None
    owner = LearningOwner(isolated_test_owner=True)
    assert owner.capability is not LearningCapabilityStatus.ACTIVE
