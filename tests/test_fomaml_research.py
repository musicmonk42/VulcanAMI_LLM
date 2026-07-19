import copy
import json

import pytest

torch = pytest.importorskip("torch")

from src.vulcan.learning.learning_types import LearningConfig
from src.vulcan.learning.meta_learning import MetaLearner, MetaLearningAlgorithm, MetaUpdateStatus


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 0.0)

    def forward(self, x):
        return self.linear(x)


def config():
    cfg = LearningConfig()
    cfg.inner_lr = 0.1
    cfg.meta_lr = 0.05
    cfg.adaptation_steps = 1
    cfg.checkpoint_frequency = 0
    return cfg


def task(slope, *, weight=1.0):
    sx = torch.tensor([[1.0], [2.0]])
    qx = torch.tensor([[3.0], [4.0]])
    return {
        "support": {"x": sx, "y": slope * sx},
        "query": {"x": qx, "y": slope * qx},
        "weight": weight,
    }


def update_for(tasks):
    model = LinearModel()
    learner = MetaLearner(model, config(), MetaLearningAlgorithm.FOMAML, rng_seed=123)
    before = model.linear.weight.detach().clone()
    result = learner.meta_update(tasks)
    return before, model.linear.weight.detach().clone(), result, learner


def delta(before, after):
    return after - before


def test_two_distinct_tasks_both_contribute_to_averaged_update():
    a = task(1.0)
    b = task(-2.0)
    ba, aa, _, _ = update_for([a])
    bb, ab, _, _ = update_for([b])
    bc, ac, result, _ = update_for([a, b])
    assert result.status is MetaUpdateStatus.APPLIED
    combined = delta(bc, ac)
    assert not torch.allclose(combined, delta(ba, aa), atol=1e-8, rtol=1e-8)
    assert not torch.allclose(combined, delta(bb, ab), atol=1e-8, rtol=1e-8)


def test_removing_either_task_and_reordering_changes_as_documented():
    a = task(1.0)
    b = task(-2.0)
    base, both, _, _ = update_for([a, b])
    _, only_a, _, _ = update_for([a])
    _, only_b, _, _ = update_for([b])
    _, reversed_both, _, _ = update_for([b, a])
    assert not torch.allclose(delta(base, both), delta(base, only_a), atol=1e-8, rtol=1e-8)
    assert not torch.allclose(delta(base, both), delta(base, only_b), atol=1e-8, rtol=1e-8)
    assert torch.allclose(both, reversed_both, atol=1e-8, rtol=1e-8)


def test_duplicate_task_changes_weighting_only_by_repetition():
    a = task(1.0)
    b = task(-2.0)
    _, one_each, r1, _ = update_for([a, b])
    _, duplicated_a, r2, _ = update_for([a, a, b])
    assert r1.task_weights == (1.0, 1.0)
    assert r2.task_weights == (1.0, 1.0, 1.0)
    assert not torch.allclose(one_each, duplicated_a, atol=1e-8, rtol=1e-8)


def test_support_adaptation_predictably_changes_query_loss():
    learner = MetaLearner(LinearModel(), config(), MetaLearningAlgorithm.FOMAML, rng_seed=9)
    t = task(1.5)
    support = learner._validate_dataset(t["support"], "support")
    query = learner._validate_dataset(t["query"], "query")
    params = {n: p.detach().clone().requires_grad_(True) for n, p in learner.base_model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in learner.base_model.named_buffers()}
    before = learner._compute_loss_with_params(params, buffers, query).detach()
    loss = learner._compute_loss_with_params(params, buffers, support)
    grads = torch.autograd.grad(loss, tuple(params.values()))
    adapted = {name: (param - learner.config.inner_lr * grad).detach().requires_grad_(True) for (name, param), grad in zip(params.items(), grads)}
    after = learner._compute_loss_with_params(adapted, buffers, query).detach()
    assert after < before


def test_missing_empty_nonfinite_data_fails_closed_and_no_dummy_generated():
    learner = MetaLearner(LinearModel(), config(), MetaLearningAlgorithm.FOMAML, rng_seed=1)
    with pytest.raises(ValueError):
        learner.meta_update([])
    with pytest.raises(ValueError):
        learner.meta_update([{"support": {"x": torch.empty(0, 1), "y": torch.empty(0, 1)}, "query": task(1.0)["query"]}])
    bad = task(1.0)
    bad["support"]["x"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        learner.meta_update([bad])
    with pytest.raises(ValueError):
        learner._create_batch_from_indices([], torch.tensor([], dtype=torch.long))


def test_save_reopen_reproduces_model_state_and_next_update(tmp_path):
    tasks = [task(1.0), task(-2.0)]
    model = LinearModel()
    learner = MetaLearner(model, config(), MetaLearningAlgorithm.FOMAML, rng_seed=2)
    learner.meta_update(tasks)
    path = tmp_path / "fomaml.json"
    learner.save_fomaml_state(str(path))
    reopened_model = LinearModel()
    reopened = MetaLearner(reopened_model, config(), MetaLearningAlgorithm.FOMAML, rng_seed=999)
    reopened.load_fomaml_state(str(path))
    assert torch.allclose(model.linear.weight, reopened_model.linear.weight, atol=0, rtol=0)
    uninterrupted = copy.deepcopy(model)
    learner2 = MetaLearner(uninterrupted, config(), MetaLearningAlgorithm.FOMAML, rng_seed=2)
    learner2.load_fomaml_state(str(path))
    learner2.meta_update(tasks)
    reopened.meta_update(tasks)
    assert torch.allclose(uninterrupted.linear.weight, reopened_model.linear.weight, atol=1e-8, rtol=1e-8)


def test_unavailable_algorithms_are_typed_and_unadvertised():
    for alg in (MetaLearningAlgorithm.MAML, MetaLearningAlgorithm.PROTO):
        learner = MetaLearner(LinearModel(), config(), alg)
        before = learner.base_model.linear.weight.detach().clone()
        result = learner.meta_update([task(1.0)])
        assert result.status is MetaUpdateStatus.UNAVAILABLE
        assert torch.equal(before, learner.base_model.linear.weight.detach())
