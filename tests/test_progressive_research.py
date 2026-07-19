import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.vulcan.learning.continual_learning import EnhancedContinualLearner
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner


def learner(seed=123):
    torch.manual_seed(seed)
    return EnhancedContinualLearner(embedding_dim=4, use_hierarchical=False, use_progressive=True, allow_unverified_progressive=True, progressive_rng_seed=seed)


def add_task(l, tid):
    l._create_new_task(tid)
    assert l.use_progressive
    return l.task_order.index(tid)


def step(l, tid, x, target=None):
    if target is None: target = x
    out = l.forward(x, tid)
    loss = torch.nn.functional.mse_loss(out, target)
    opt = l.optimizers[tid]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for g in opt.param_groups for p in g["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return float(loss.detach()), float(norm.detach() if hasattr(norm, "detach") else norm)


def state_bytes(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def different(before, after, prefix):
    return any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))


def test_progressive_column_and_lateral_update_and_prior_frozen():
    l = learner(7)
    c0 = add_task(l, "task_a")
    c1 = add_task(l, "task_b")
    assert c0 == 0 and c1 == 1
    before = state_bytes(l.progressive_network)
    x = torch.tensor([[0.2, -0.1, 0.3, 0.5]], dtype=torch.float32)
    loss, grad_norm = step(l, "task_b", x, torch.zeros_like(x))
    after = state_bytes(l.progressive_network)
    assert loss >= 0 and grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))
    assert different(before, after, "columns.1.")
    assert different(before, after, "lateral_connections.0_to_1.")
    for k in before:
        if k.startswith("columns.0."):
            assert torch.equal(before[k], after[k])


def test_unowned_parameter_gradients_do_not_accumulate():
    l = learner(8); add_task(l, "a"); add_task(l, "b")
    x = torch.ones(1, 4)
    step(l, "b", x, torch.zeros_like(x))
    for name, p in l.progressive_network.named_parameters():
        if name.startswith("columns.0."):
            assert p.grad is None or torch.count_nonzero(p.grad).item() == 0


def test_lateral_ablation_changes_final_output_for_nonzero_fixture():
    l = learner(9); add_task(l, "a"); add_task(l, "b")
    x = torch.randn(2, 4)
    full = l.progressive_network(x, 1)
    ablated = l.progressive_network(x, 1, ablate_lateral=True)
    assert not torch.allclose(full, ablated)


def test_same_seed_and_data_produce_same_update():
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    a = learner(10); b = learner(10)
    add_task(a, "a"); add_task(a, "b")
    add_task(b, "a"); add_task(b, "b")
    step(a, "b", x); step(b, "b", x)
    for (ka, va), (kb, vb) in zip(a.progressive_network.state_dict().items(), b.progressive_network.state_dict().items()):
        assert ka == kb
        assert torch.allclose(va, vb, atol=0, rtol=0)


def test_save_reopen_identical_eval_and_next_step(tmp_path):
    x = torch.tensor([[0.4, -0.2, 0.7, 0.1]])
    l = learner(11); add_task(l, "a"); add_task(l, "b")
    step(l, "b", x, torch.zeros_like(x))
    expected = l.forward(x, "b").detach().clone()
    path = tmp_path / "progressive.json"
    l.save_progressive_research_state(str(path))
    reopened = learner(999)
    reopened.load_progressive_research_state(str(path))
    assert torch.allclose(reopened.forward(x, "b"), expected, atol=0, rtol=0)
    l2 = copy.deepcopy(l)
    step(l, "b", x, torch.ones_like(x))
    step(reopened, "b", x, torch.ones_like(x))
    for k, v in l.progressive_network.state_dict().items():
        assert torch.allclose(v, reopened.progressive_network.state_dict()[k], atol=1e-7, rtol=1e-7)
    assert any(not torch.equal(l2.progressive_network.state_dict()[k], l.progressive_network.state_dict()[k]) for k in l.progressive_network.state_dict())


def test_corrupt_incomplete_reordered_dimension_mismatch_state_fails(tmp_path):
    l = learner(12); add_task(l, "a"); path = tmp_path / "s.json"; l.save_progressive_research_state(str(path))
    doc = json.loads(path.read_text())
    doc["embedding_dim"] = 5
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError): learner(12).load_progressive_research_state(str(bad))
    doc = json.loads(path.read_text()); doc.pop("task_order")
    bad2 = tmp_path / "bad2.json"; bad2.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError): learner(12).load_progressive_research_state(str(bad2))


def test_default_production_remains_progressive_disabled_and_unadvertised():
    l = EnhancedContinualLearner(use_hierarchical=False)
    assert l.use_progressive is False
    assert not hasattr(l, "progressive_network")
    owner = LearningOwner(isolated_test_owner=True)
    assert owner.capability is not LearningCapabilityStatus.ACTIVE
