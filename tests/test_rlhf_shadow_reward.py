import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.vulcan.learning.rlhf_feedback import (
    FeedbackIngestionStatus,
    RLHFManager,
    ShadowRewardModelTrainer,
    ShadowRewardTrainingStatus,
    TrustedFeatureBatch,
    build_evidence_bound_feedback,
)

HEX = "0" * 64


class FakeEncoder:
    encoder_id = "fake-separable"
    release_digest = hashlib.sha256(b"fake-separable-v1").hexdigest()
    output_dim = 2

    def encode(self, normalized_input: str) -> TrustedFeatureBatch:
        if normalized_input == "bad":
            return TrustedFeatureBatch(torch.tensor([float("nan"), 0.0]), self.release_digest)
        sign = 1.0 if "good" in normalized_input else -1.0
        features = torch.tensor([sign, 0.0], dtype=torch.float32)
        return TrustedFeatureBatch(features, self.release_digest)


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def feedback(label, reward, *, case=None, ledger=None, obs=None):
    return build_evidence_bound_feedback(
        observation_id=obs or f"obs-{label}",
        case_digest=case or digest(f"case-{label}"),
        request_digest=digest(f"request-{label}"),
        tenant_digest=digest("tenant-a"),
        response_ledger_digest=ledger or digest(f"ledger-{label}"),
        reviewer_digest=digest("reviewer"),
        reviewer_verifier_digest=digest("verifier"),
        reward=reward,
        normalized_input="good item" if reward > 0 else "bad item",
        timestamp_utc_microseconds=1_700_000_000_000_000 + len(label),
        encoder_release_digest=FakeEncoder.release_digest,
    )


def case_for_bucket(prefix, target_bucket):
    # Trainer split: <6 train, <8 validation, else heldout.
    for i in range(10000):
        d = digest(f"{prefix}-{i}")
        bucket = int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) % 10
        name = "train" if bucket < 6 else "validation" if bucket < 8 else "heldout"
        if name == target_bucket:
            return d
    raise AssertionError("bucket not found")


def populated_trainer():
    trainer = ShadowRewardModelTrainer(FakeEncoder(), 2, seed=7, capacity=32, min_examples=6)
    rows = []
    for bucket in ("train", "validation", "heldout"):
        rows.append(feedback(f"{bucket}-pos", 1.0, case=case_for_bucket(f"{bucket}-pos", bucket)))
        rows.append(feedback(f"{bucket}-neg", -1.0, case=case_for_bucket(f"{bucket}-neg", bucket)))
    for row in rows:
        assert trainer.submit_feedback(row).status in (FeedbackIngestionStatus.ACCEPTED, FeedbackIngestionStatus.DUPLICATE)
    return trainer


def test_identical_text_and_release_produce_byte_identical_features():
    trainer = ShadowRewardModelTrainer(FakeEncoder(), 2)
    a = trainer.encode_features("good item")
    b = trainer.encode_features("good item")
    assert a.numpy().tobytes() == b.numpy().tobytes()


def test_restart_process_produces_identical_features(tmp_path):
    script = tmp_path / "check.py"
    script.write_text(
        "import hashlib, torch\n"
        "from src.vulcan.learning.rlhf_feedback import TrustedFeatureBatch, ShadowRewardModelTrainer\n"
        "class E:\n encoder_id='fake'; release_digest=hashlib.sha256(b'fake-separable-v1').hexdigest(); output_dim=2\n def encode(self, s): return TrustedFeatureBatch(torch.tensor([1.0,0.0]), self.release_digest)\n"
        "print(ShadowRewardModelTrainer(E(),2).encode_features('good item').numpy().tobytes().hex())\n"
    )
    out1 = subprocess.check_output([sys.executable, str(script)], cwd=Path.cwd()).strip()
    out2 = subprocess.check_output([sys.executable, str(script)], cwd=Path.cwd()).strip()
    assert out1 == out2


def test_unavailable_unknown_encoder_and_random_fallback_fail_closed():
    with pytest.raises(RuntimeError):
        ShadowRewardModelTrainer(None, 2)
    manager = RLHFManager(torch.nn.Linear(2, 2), feature_encoder=None)
    try:
        with pytest.raises(RuntimeError):
            manager._extract_features("unencoded text")
    finally:
        manager.shutdown()


def test_duplicate_feedback_one_effect_and_altered_binding_fails():
    trainer = ShadowRewardModelTrainer(FakeEncoder(), 2, capacity=4)
    item = feedback("dup", 1.0)
    assert trainer.submit_feedback(item).status is FeedbackIngestionStatus.ACCEPTED
    assert trainer.submit_feedback(item).status is FeedbackIngestionStatus.DUPLICATE
    assert trainer.pending_count == 1
    altered = feedback("dup2", -1.0, obs=item.observation_id)
    with pytest.raises(ValueError):
        trainer.submit_feedback(altered)
    with pytest.raises(ValueError):
        build_evidence_bound_feedback(
            observation_id="x", case_digest=HEX, request_digest=HEX, tenant_digest=HEX,
            response_ledger_digest=HEX, reviewer_digest=HEX, reviewer_verifier_digest=HEX,
            reward=float("nan"), normalized_input="good", timestamp_utc_microseconds=1,
            encoder_release_digest=FakeEncoder.release_digest,
        )


def test_shadow_training_improves_heldout_over_baseline_and_is_not_active(tmp_path):
    trainer = populated_trainer()
    result = trainer.train_shadow_candidate(epochs=60, threshold=0.2)
    assert result.status is ShadowRewardTrainingStatus.CANDIDATE_READY
    assert result.heldout_metric >= result.baseline_metric + result.threshold
    assert trainer.predict_shadow("good item") > trainer.predict_shadow("bad item")
    manager = RLHFManager(torch.nn.Linear(2, 2), feature_encoder=FakeEncoder())
    before_policy = {k: v.detach().clone() for k, v in manager.base_model.state_dict().items()}
    try:
        assert manager.update_policy_with_ppo([])["status"] == "unavailable"
        for key, value in manager.base_model.state_dict().items():
            assert torch.equal(value, before_policy[key])
    finally:
        manager.shutdown()
    path = tmp_path / "candidate.json"
    trainer.save_shadow_candidate(str(path))
    reopened = ShadowRewardModelTrainer(FakeEncoder(), 2)
    reopened.load_shadow_candidate(str(path))
    assert abs(trainer.predict_shadow("good item") - reopened.predict_shadow("good item")) <= 1e-7


def test_training_only_improvement_with_missing_heldout_is_rejected():
    trainer = ShadowRewardModelTrainer(FakeEncoder(), 2, min_examples=2)
    trainer.submit_feedback(feedback("train-pos", 1.0, case=case_for_bucket("train-pos", "train")))
    trainer.submit_feedback(feedback("train-neg", -1.0, case=case_for_bucket("train-neg", "train")))
    result = trainer.train_shadow_candidate()
    assert result.status is ShadowRewardTrainingStatus.REJECTED


def test_repeated_seeded_training_same_candidate_digest_and_bounded_close():
    a = populated_trainer(); b = populated_trainer()
    ra = a.train_shadow_candidate(); rb = b.train_shadow_candidate()
    assert ra.candidate_digest == rb.candidate_digest
    a.close(); a.close()
    assert a.pending_count == 0
    with pytest.raises(RuntimeError):
        a.submit_feedback(feedback("closed", 1.0))
