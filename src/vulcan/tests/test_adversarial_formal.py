# test_adversarial_formal.py
"""
Comprehensive test suite for adversarial validation and formal verification.
Tests AdversarialValidator and FormalVerifier classes.
"""

from vulcan.safety.safety_types import (ActionType, SafetyReport,
                                        SafetyViolationType)
from vulcan.safety.adversarial_formal import (AdversarialValidator,
                                              AttackConfig, AttackType,
                                              FormalVerifier, timeout)
import copy
import time

import numpy as np
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip(
    "torch", reason="PyTorch required for adversarial_formal tests"
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""
    return {
        "type": ActionType.EXPLORE,
        "confidence": 0.85,
        "embedding": np.random.randn(64).tolist(),
        "resource_usage": {"cpu": 0.5, "memory": 0.3},
        "safe": True,
    }


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return {
        "resource_limits": {"cpu": 1.0, "memory": 1.0},
        "consistent": True,
        "timestamp": time.time(),
    }


@pytest.fixture
def adversarial_validator():
    """Create an AdversarialValidator instance."""
    return AdversarialValidator(epsilon=0.1, num_attacks=5, random_seed=42)


@pytest.fixture
def formal_verifier():
    """Create a FormalVerifier instance."""
    return FormalVerifier()


@pytest.fixture
def custom_validator():
    """Create a custom validator function for testing."""

    def validator(action, context):
        if action.get("backdoor"):
            return False, "Backdoor detected", 0.9
        if action.get("confidence", 0) < 0 or action.get("confidence", 1) > 1:
            return False, "Invalid confidence", 0.8
        return True, "OK", 0.9

    return validator


# ============================================================
# TIMEOUT TESTS
# ============================================================


class TestTimeout:
    """Test timeout context manager."""

    def test_timeout_normal_execution(self):
        """Test that normal execution completes without timeout."""
        with timeout(2):
            time.sleep(0.1)
            result = 42
        assert result == 42

    def test_timeout_raises_on_exceed(self):
        """Test that timeout raises TimeoutError when exceeded."""
        # Only test on Unix systems with SIGALRM
        import signal

        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")

        with pytest.raises(TimeoutError):
            with timeout(1):
                time.sleep(2)

    def test_timeout_windows_compatibility(self):
        """Test timeout works on Windows (returns checker object)."""
        import signal

        # Test actual Windows behavior if on Windows
        if not hasattr(signal, "SIGALRM"):
            # We're on Windows - test the actual Windows behavior
            with timeout(2) as checker:
                # On Windows, should get a TimeoutChecker object
                assert checker is not None
                assert hasattr(checker, "check")

                # Should be able to check timeout status
                time.sleep(0.1)
                checker.check()  # Should not raise
        else:
            # We're on Unix - skip this test as it's Windows-specific
            pytest.skip("Test is for Windows platform only")


# ============================================================
# ADVERSARIAL VALIDATOR TESTS
# ============================================================


class TestAdversarialValidator:
    """Test AdversarialValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = AdversarialValidator(epsilon=0.2, num_attacks=10)
        assert validator.epsilon == 0.2
        assert validator.num_attacks == 10
        assert len(validator.attack_methods) > 0
        assert len(validator.attack_configs) > 0

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        action = {"type": ActionType.EXPLORE, "embedding": [0.5] * 64}
        context = {}

        validator1 = AdversarialValidator(epsilon=0.1, num_attacks=3, random_seed=42)
        result1 = validator1._fgsm_attack(
            action, context, validator1.default_attack_config
        )

        validator2 = AdversarialValidator(epsilon=0.1, num_attacks=3, random_seed=42)
        result2 = validator2._fgsm_attack(
            action, context, validator2.default_attack_config
        )

        # Results should be identical with same seed
        assert len(result1) == len(result2)
        if result1:
            assert np.allclose(
                result1[0]["embedding"], result2[0]["embedding"], atol=1e-6
            )

    def test_validate_robustness_basic(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test basic robustness validation."""
        report = adversarial_validator.validate_robustness(
            sample_action, sample_context, attack_types=[AttackType.FGSM]
        )

        assert isinstance(report, SafetyReport)
        assert isinstance(report.safe, bool)
        assert 0 <= report.confidence <= 1
        assert "robustness_score" in report.metadata
        assert "attack_results" in report.metadata

    def test_validate_robustness_with_custom_validator(
        self, adversarial_validator, sample_action, sample_context, custom_validator
    ):
        """Test robustness validation with custom validator."""
        report = adversarial_validator.validate_robustness(
            sample_action,
            sample_context,
            validator=custom_validator,
            attack_types=[AttackType.FGSM, AttackType.SEMANTIC],
        )

        assert isinstance(report, SafetyReport)
        assert "attacks_tested" in report.metadata
        assert len(report.metadata["attacks_tested"]) >= 1

    def test_validate_robustness_timeout(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test that timeout protection works."""

        # Mock an attack that would hang
        def slow_attack(*args, **kwargs):
            time.sleep(10)  # Would timeout
            return []

        original_method = adversarial_validator.attack_methods[AttackType.FGSM]
        adversarial_validator.attack_methods[AttackType.FGSM] = slow_attack

        try:
            report = adversarial_validator.validate_robustness(
                sample_action,
                sample_context,
                attack_types=[AttackType.FGSM],
                timeout_per_attack=1.0,
            )

            # Should complete with timeout message
            assert "timed out" in " ".join(report.reasons).lower() or report.safe
        finally:
            # Restore original method
            adversarial_validator.attack_methods[AttackType.FGSM] = original_method

    def test_fgsm_attack(self, adversarial_validator, sample_action, sample_context):
        """Test FGSM attack generation."""
        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._fgsm_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("adversarial") for p in perturbed)
        assert all(p.get("attack_type") == "fgsm" for p in perturbed)

        # Check perturbation magnitude
        if perturbed and "embedding" in perturbed[0]:
            original_emb = np.array(sample_action["embedding"])
            perturbed_emb = np.array(perturbed[0]["embedding"])
            perturbation = np.linalg.norm(perturbed_emb - original_emb)
            assert perturbation > 0  # Some perturbation applied

    def test_pgd_attack(self, adversarial_validator, sample_action, sample_context):
        """Test PGD attack generation."""
        config = AttackConfig(epsilon=0.1, alpha=0.025, num_iterations=10)
        perturbed = adversarial_validator._pgd_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "pgd" for p in perturbed)
        assert all("iterations" in p for p in perturbed)

    def test_semantic_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test semantic attack generation."""
        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._semantic_attack(
            sample_action, sample_context, config
        )

        # Should generate type substitutions
        assert len(perturbed) > 0
        type_changes = [
            p for p in perturbed if p.get("type") != sample_action.get("type")
        ]
        assert len(type_changes) > 0

    def test_semantic_attack_text(self, adversarial_validator, sample_context):
        """Test semantic attack on text fields."""
        action = {
            "type": ActionType.EXPLORE,
            "text": "This action is safe and will help the user.",
        }

        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._semantic_attack(
            action, sample_context, config
        )

        # Should substitute words
        text_changes = [p for p in perturbed if p.get("text"] != action.get("text"))
        assert len(text_changes) > 0

    def test_boundary_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test boundary attack generation."""
        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._boundary_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0

        # Check for extreme values
        confidence_values = [
            p.get("confidence") for p in perturbed if "confidence" in p
        ]
        if confidence_values:
            assert any(v in [0.0, 0.01, 0.99, 1.0] for v in confidence_values)

    def test_boundary_attack_overflow(self, adversarial_validator, sample_context):
        """Test boundary attack with integer overflow values."""
        action = {"type": ActionType.EXPLORE, "count": 100}

        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._boundary_attack(
            action, sample_context, config
        )

        overflow_values = [p.get("count") for p in perturbed if "count" in p]
        assert any(v in [-1, 0, 2**31 - 1] for v in overflow_values)

    def test_trojan_attack(self, adversarial_validator, sample_action, sample_context):
        """Test trojan attack generation."""
        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._trojan_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0

        # Check for backdoor triggers
        backdoor_present = any(
            "backdoor" in p or "BACKDOOR" in str(p) or "bypass_security" in p
            for p in perturbed
        )
        assert backdoor_present

    def test_deepfool_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test DeepFool attack generation."""
        config = AttackConfig(epsilon=0.1, max_iterations=20, early_stop=True)
        perturbed = adversarial_validator._deepfool_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "deepfool" for p in perturbed)
        assert all("iterations_used" in p for p in perturbed)

    def test_deepfool_convergence(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test DeepFool convergence protection."""
        config = AttackConfig(epsilon=0.1, max_iterations=100, early_stop=True)

        start_time = time.time()
        perturbed = adversarial_validator._deepfool_attack(
            sample_action, sample_context, config
        )
        elapsed = time.time() - start_time

        # Should complete quickly due to convergence checks
        assert elapsed < 5.0  # Should not take too long
        assert len(perturbed) > 0

    def test_carlini_wagner_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test Carlini & Wagner attack generation."""
        config = AttackConfig(epsilon=0.1, max_iterations=50, alpha=0.01)
        perturbed = adversarial_validator._carlini_wagner_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "carlini_wagner" for p in perturbed)
        assert all("final_const" in p for p in perturbed)

    def test_jsma_attack(self, adversarial_validator, sample_action, sample_context):
        """Test JSMA attack generation."""
        config = AttackConfig(epsilon=0.1, max_iterations=50)
        perturbed = adversarial_validator._jsma_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "jsma" for p in perturbed)
        assert all("features_modified" in p for p in perturbed)

    def test_universal_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test universal adversarial perturbation."""
        config = AttackConfig(epsilon=0.1)
        perturbed = adversarial_validator._universal_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "universal" for p in perturbed)
        assert all("perturbation_scale" in p for p in perturbed)

    def test_targeted_attack(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test targeted misclassification attack."""
        config = AttackConfig(epsilon=0.1, targeted=True)
        perturbed = adversarial_validator._targeted_attack(
            sample_action, sample_context, config
        )

        assert len(perturbed) > 0
        assert all(p.get("attack_type") == "targeted" for p in perturbed)
        assert all("target_class" in p for p in perturbed)

    def test_default_validate(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test default validation function."""
        safe, reason, confidence = adversarial_validator._default_validate(
            sample_action, sample_context
        )

        assert isinstance(safe, bool)
        assert isinstance(reason, str)
        assert 0 <= confidence <= 1

    def test_default_validate_detects_adversarial(
        self, adversarial_validator, sample_context
    ):
        """Test that default validator can detect adversarial patterns."""
        adversarial_action = {
            "type": ActionType.EXPLORE,
            "adversarial": True,
            "attack_type": "fgsm",
            "embedding": [0.5] * 64,
        }

        # Run multiple times since detection is probabilistic
        detections = []
        for _ in range(10):
            safe, reason, confidence = adversarial_validator._default_validate(
                adversarial_action, sample_context
            )
            detections.append(not safe)

        # Should detect at least some
        assert any(detections)

    def test_default_validate_detects_malicious(
        self, adversarial_validator, sample_context
    ):
        """Test that default validator detects malicious patterns."""
        malicious_action = {
            "type": ActionType.EXPLORE,
            "backdoor": True,
            "bypass_security": True,
        }

        safe, reason, confidence = adversarial_validator._default_validate(
            malicious_action, sample_context
        )

        assert not safe
        assert "malicious" in reason.lower()

    def test_calculate_perturbation_norm_embedding(self, adversarial_validator):
        """Test perturbation norm calculation for embeddings."""
        original = {"embedding": [1.0, 2.0, 3.0]}
        perturbed = {"embedding": [1.1, 2.1, 3.1]}

        norm = adversarial_validator._calculate_perturbation_norm(original, perturbed)
        expected = np.linalg.norm(np.array([0.1, 0.1, 0.1]))
        assert abs(norm - expected) < 1e-6

    def test_calculate_perturbation_norm_fields(self, adversarial_validator):
        """Test perturbation norm calculation for changed fields."""
        original = {"type": "A", "confidence": 0.5, "safe": True}
        perturbed = {"type": "B", "confidence": 0.5, "safe": False}

        norm = adversarial_validator._calculate_perturbation_norm(original, perturbed)
        assert norm == 2.0  # Two fields changed

    def test_calculate_robustness_score(self, adversarial_validator):
        """Test robustness score calculation."""
        attack_results = [
            {"attack_type": "fgsm", "successful": False},
            {"attack_type": "pgd", "successful": False},
            {"attack_type": "semantic", "successful": True},
            {"attack_type": "boundary", "successful": False},
        ]

        score = adversarial_validator._calculate_robustness_score(attack_results)
        assert 0 <= score <= 1
        assert score > 0.5  # Mostly successful defense

    def test_calculate_robustness_score_all_fail(self, adversarial_validator):
        """Test robustness score when all attacks succeed."""
        attack_results = [
            {"attack_type": "fgsm", "successful": True},
            {"attack_type": "pgd", "successful": True},
        ]

        score = adversarial_validator._calculate_robustness_score(attack_results)
        assert score < 0.5  # Poor robustness

    def test_identify_vulnerability_patterns(self, adversarial_validator):
        """Test vulnerability pattern identification."""
        violations = [SafetyViolationType.ADVERSARIAL]
        attack_results = [
            {"attack_type": "fgsm", "successful": True, "perturbation_norm": 0.05},
            {"attack_type": "pgd", "successful": True, "perturbation_norm": 0.08},
            {"attack_type": "semantic", "successful": True},
            {"attack_type": "trojan", "successful": True},
            {"attack_type": "boundary", "successful": True},
        ]

        patterns = adversarial_validator._identify_vulnerability_patterns(
            violations, attack_results
        )

        assert len(patterns) > 0
        assert any("multiple attack types" in p.lower() for p in patterns)

    def test_suggest_mitigations(self, adversarial_validator):
        """Test mitigation suggestion."""
        violations = [SafetyViolationType.ADVERSARIAL]
        patterns = ["Vulnerable to small perturbations", "Weak input validation"]

        mitigations = adversarial_validator._suggest_mitigations(violations, patterns)

        assert len(mitigations) > 0
        assert any("adversarial training" in m.lower() for m in mitigations)
        assert any("input validation" in m.lower() for m in mitigations)

    def test_record_attack(self, adversarial_validator, sample_action):
        """Test attack recording."""
        perturbed = copy.deepcopy(sample_action)
        perturbed["confidence"] = 0.9

        initial_len = len(adversarial_validator.attack_history)

        adversarial_validator._record_attack(
            AttackType.FGSM,
            sample_action,
            perturbed,
            success=True,
            reason="Test attack",
        )

        assert len(adversarial_validator.attack_history) == initial_len + 1
        assert adversarial_validator.attack_metrics[AttackType.FGSM]["successes"] > 0

    def test_get_attack_stats(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test getting attack statistics."""
        # Generate some attacks
        adversarial_validator.validate_robustness(
            sample_action,
            sample_context,
            attack_types=[AttackType.FGSM, AttackType.SEMANTIC],
        )

        stats = adversarial_validator.get_attack_stats()

        assert "total_attacks" in stats
        assert "attack_types" in stats
        assert isinstance(stats["attack_types"], dict)

    def test_thread_safety(self, adversarial_validator, sample_action, sample_context):
        """Test thread-safe operations."""
        import threading

        results = []

        def run_validation():
            report = adversarial_validator.validate_robustness(
                sample_action, sample_context, attack_types=[AttackType.FGSM]
            )
            results.append(report)

        threads = [threading.Thread(target=run_validation) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, SafetyReport) for r in results)


# ============================================================
# FORMAL VERIFIER TESTS
# ============================================================


class TestFormalVerifier:
    """Test FormalVerifier class."""

    def test_initialization(self):
        """Test verifier initialization."""
        verifier = FormalVerifier()
        assert len(verifier.properties) > 0  # Has default properties
        assert len(verifier.verification_methods) > 0

    def test_add_safety_property(self, formal_verifier):
        """Test adding safety property."""
        initial_count = len(formal_verifier.properties)

        formal_verifier.add_safety_property(
            lambda a, s: a.get("safe", True),
            "test_property",
            "Test property description",
        )

        assert len(formal_verifier.properties) == initial_count + 1
        assert any(p.name == "test_property" for p in formal_verifier.properties)

    def test_add_invariant(self, formal_verifier):
        """Test adding invariant."""
        initial_count = len(formal_verifier.invariants)

        formal_verifier.add_invariant(
            lambda s: s.get("valid", True),
            "test_invariant",
            "Test invariant description",
        )

        assert len(formal_verifier.invariants) == initial_count + 1
        assert any(i.name == "test_invariant" for i in formal_verifier.invariants)

    def test_add_temporal_property(self, formal_verifier):
        """Test adding temporal property."""
        initial_count = len(formal_verifier.temporal_properties)

        formal_verifier.add_temporal_property(
            lambda history: all(h[0].get("safe", True) for h in history),
            "test_temporal",
            "Test temporal property",
            window=5,
        )

        assert len(formal_verifier.temporal_properties) == initial_count + 1

    def test_add_probabilistic_property(self, formal_verifier):
        """Test adding probabilistic property."""
        initial_count = len(formal_verifier.probabilistic_properties)

        formal_verifier.add_probabilistic_property(
            lambda s: 0.9,
            "test_probabilistic",
            "Test probabilistic property",
            threshold=0.8,
        )

        assert len(formal_verifier.probabilistic_properties) == initial_count + 1

    def test_verify_action_model_checking(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test action verification using model checking."""
        report = formal_verifier.verify_action(
            sample_action, sample_context, method="model_checking"
        )

        assert isinstance(report, SafetyReport)
        assert isinstance(report.safe, bool)
        assert "verification_method" in report.metadata
        assert report.metadata["verification_method"] == "model_checking"

    def test_verify_action_theorem_proving(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test action verification using theorem proving."""
        report = formal_verifier.verify_action(
            sample_action, sample_context, method="theorem_proving"
        )

        assert isinstance(report, SafetyReport)
        assert report.metadata["verification_method"] == "theorem_proving"
        assert "theorems_proved" in report.metadata

    def test_verify_action_symbolic_execution(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test action verification using symbolic execution."""
        report = formal_verifier.verify_action(
            sample_action, sample_context, method="symbolic_execution"
        )

        assert isinstance(report, SafetyReport)
        assert report.metadata["verification_method"] == "symbolic_execution"
        assert "paths_explored" in report.metadata

    def test_verify_action_abstract_interpretation(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test action verification using abstract interpretation."""
        report = formal_verifier.verify_action(
            sample_action, sample_context, method="abstract_interpretation"
        )

        assert isinstance(report, SafetyReport)
        assert report.metadata["verification_method"] == "abstract_interpretation"

    def test_verify_action_caching(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test that verification results are cached."""
        # First call
        report1 = formal_verifier.verify_action(sample_action, sample_context)
        cache_size1 = len(formal_verifier.verification_cache)

        # Second call with same inputs
        report2 = formal_verifier.verify_action(sample_action, sample_context)
        cache_size2 = len(formal_verifier.verification_cache)

        assert cache_size2 == cache_size1  # No new cache entry
        assert report1.safe == report2.safe

    def test_verify_action_cache_limit(self, formal_verifier, sample_context):
        """Test that cache enforces size limit."""
        formal_verifier.cache_max_size = 10

        # Generate many different actions
        for i in range(20):
            action = {"type": ActionType.EXPLORE, "id": i}
            formal_verifier.verify_action(action, sample_context)

        # Cache should be limited
        assert len(formal_verifier.verification_cache) <= formal_verifier.cache_max_size

    def test_property_violation_detection(self, formal_verifier, sample_context):
        """Test detection of property violations."""
        # Add a property that will be violated
        formal_verifier.add_safety_property(
            lambda a, s: a.get("confidence", 0) > 0.5,
            "min_confidence",
            "Confidence must be above 0.5",
        )

        # Action that violates property
        unsafe_action = {
            "type": ActionType.EXPLORE,
            "confidence": 0.3,  # Below threshold
            "safe": True,
        }

        report = formal_verifier.verify_action(unsafe_action, sample_context)

        assert not report.safe
        assert any("min_confidence" in r for r in report.reasons)

    def test_invariant_checking(self, formal_verifier, sample_action):
        """Test invariant checking."""
        # Add an invariant that will be violated
        formal_verifier.add_invariant(
            lambda s: s.get("consistent", False),
            "consistency_check",
            "State must be consistent",
        )

        inconsistent_state = {"consistent": False}

        report = formal_verifier.verify_action(sample_action, inconsistent_state)

        assert not report.safe
        assert any("consistency_check" in r for r in report.reasons)

    def test_temporal_property_checking(self, formal_verifier, sample_context):
        """Test temporal property checking."""
        # Add temporal property
        formal_verifier.add_temporal_property(
            lambda history: len(history) <= 5,
            "max_sequence_length",
            "Sequence length must not exceed 5",
            window=10,
        )

        action = {"type": ActionType.EXPLORE, "safe": True}

        # Verify multiple times to build history
        for _ in range(12):
            report = formal_verifier.verify_action(action, sample_context)

        # Temporal property should be checked
        assert isinstance(report, SafetyReport)

    def test_probabilistic_property_checking(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test probabilistic property checking."""
        # Add probabilistic property
        formal_verifier.add_probabilistic_property(
            lambda s: 0.95,
            "high_confidence",
            "Should have high confidence",
            threshold=0.9,
        )

        # Verify multiple times to collect samples
        for _ in range(15):
            report = formal_verifier.verify_action(sample_action, sample_context)

        assert isinstance(report, SafetyReport)

    def test_counterexample_generation(self, formal_verifier, sample_context):
        """Test counterexample generation on violation."""
        # Add property that will be violated
        formal_verifier.add_safety_property(
            lambda a, s: False,  # Always fails
            "always_fail",
            "This property always fails",
        )

        action = {"type": ActionType.EXPLORE}

        initial_counterexamples = len(formal_verifier.counterexamples)
        report = formal_verifier.verify_action(action, sample_context)

        assert not report.safe
        assert len(formal_verifier.counterexamples) > initial_counterexamples

    def test_generate_proof_certificate_success(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test proof certificate generation for verified action."""
        certificate = formal_verifier.generate_proof_certificate(
            sample_action, sample_context
        )

        if certificate:  # May be None if verification fails
            assert "certificate_id" in certificate
            assert "timestamp" in certificate
            assert "action_hash" in certificate
            assert "signature" in certificate
            assert "properties_verified" in certificate

    def test_generate_proof_certificate_failure(self, formal_verifier, sample_context):
        """Test that no certificate is generated for unsafe action."""
        # Add property that will fail
        formal_verifier.add_safety_property(
            lambda a, s: False, "fail_property", "Always fails"
        )

        action = {"type": ActionType.EXPLORE}
        certificate = formal_verifier.generate_proof_certificate(action, sample_context)

        assert certificate is None

    def test_get_verification_stats(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test getting verification statistics."""
        # Run some verifications
        for _ in range(5):
            formal_verifier.verify_action(sample_action, sample_context)

        stats = formal_verifier.get_verification_stats()

        assert "properties" in stats
        assert "invariants" in stats
        assert "total_verifications" in stats
        assert stats["total_verifications"] > 0

    def test_property_priority_ordering(self, formal_verifier, sample_context):
        """Test that properties are checked in priority order."""
        # Add properties with different priorities
        formal_verifier.add_safety_property(
            lambda a, s: True, "low_priority", "Low priority property", priority=1
        )
        formal_verifier.add_safety_property(
            lambda a, s: True, "high_priority", "High priority property", priority=10
        )

        action = {"type": ActionType.EXPLORE}
        report = formal_verifier.verify_action(action, sample_context)

        # Verification should complete successfully
        assert isinstance(report, SafetyReport)

    def test_thread_safety(self, formal_verifier, sample_action, sample_context):
        """Test thread-safe operations."""
        import threading

        results = []

        def run_verification():
            report = formal_verifier.verify_action(sample_action, sample_context)
            results.append(report)

        threads = [threading.Thread(target=run_verification) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, SafetyReport) for r in results)

    def test_abstract_state_creation(self, formal_verifier):
        """Test abstract state creation."""
        state = {"confidence": 0.85, "risk": 0.2, "status": "active"}

        abstract = formal_verifier._abstract_state(state)

        assert isinstance(abstract, dict)
        assert "confidence" in abstract
        assert isinstance(abstract["confidence"], dict)
        assert "min" in abstract["confidence"]
        assert "max" in abstract["confidence"]

    def test_abstract_action_creation(self, formal_verifier):
        """Test abstract action creation."""
        action = {"type": ActionType.EXPLORE, "confidence": 0.85, "safe": True}

        abstract = formal_verifier._abstract_action(action)

        assert isinstance(abstract, dict)
        assert "confidence" in abstract
        assert "safe" in abstract


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for combined usage."""

    def test_adversarial_validation_with_formal_verification(
        self, adversarial_validator, formal_verifier, sample_action, sample_context
    ):
        """Test using both adversarial validation and formal verification."""
        # First, formal verification
        formal_report = formal_verifier.verify_action(sample_action, sample_context)

        # Then, adversarial validation
        adv_report = adversarial_validator.validate_robustness(
            sample_action,
            sample_context,
            attack_types=[AttackType.FGSM, AttackType.SEMANTIC],
        )

        # Both should provide reports
        assert isinstance(formal_report, SafetyReport)
        assert isinstance(adv_report, SafetyReport)

    def test_full_safety_pipeline(
        self, adversarial_validator, formal_verifier, sample_action, sample_context
    ):
        """Test complete safety validation pipeline."""
        # Step 1: Formal verification
        formal_report = formal_verifier.verify_action(sample_action, sample_context)

        if not formal_report.safe:
            # Action failed formal verification
            assert len(formal_report.violations) > 0
            return

        # Step 2: Adversarial validation
        adv_report = adversarial_validator.validate_robustness(
            sample_action,
            sample_context,
            attack_types=[AttackType.FGSM, AttackType.PGD, AttackType.SEMANTIC],
        )

        # Analyze combined results
        overall_safe = formal_report.safe and adv_report.safe
        combined_confidence = min(formal_report.confidence, adv_report.confidence)

        assert isinstance(overall_safe, bool)
        assert 0 <= combined_confidence <= 1


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_action(self, adversarial_validator, sample_context):
        """Test handling of empty action."""
        empty_action = {}

        report = adversarial_validator.validate_robustness(
            empty_action, sample_context, attack_types=[AttackType.FGSM]
        )

        assert isinstance(report, SafetyReport)

    def test_none_embedding(self, adversarial_validator, sample_context):
        """Test handling of None embedding."""
        action = {"type": ActionType.EXPLORE, "embedding": None}

        report = adversarial_validator.validate_robustness(
            action, sample_context, attack_types=[AttackType.FGSM]
        )

        assert isinstance(report, SafetyReport)

    def test_invalid_attack_type(
        self, adversarial_validator, sample_action, sample_context
    ):
        """Test handling of invalid attack type."""

        # Should skip invalid attack types
        class FakeAttackType:
            value = "fake_attack"

        report = adversarial_validator.validate_robustness(
            sample_action, sample_context, attack_types=[FakeAttackType()]
        )

        assert isinstance(report, SafetyReport)

    def test_property_exception_handling(
        self, formal_verifier, sample_action, sample_context
    ):
        """Test handling of exceptions in property checking."""

        # Add property that raises exception
        def faulty_property(action, state):
            raise ValueError("Test exception")

        formal_verifier.add_safety_property(
            faulty_property, "faulty_property", "This property raises an exception"
        )

        report = formal_verifier.verify_action(sample_action, sample_context)

        # Should handle exception gracefully
        assert isinstance(report, SafetyReport)
        assert not report.safe
        assert any("failed" in r.lower() for r in report.reasons)

    def test_zero_epsilon_attack(self):
        """Test attack with zero epsilon."""
        validator = AdversarialValidator(epsilon=0.0, num_attacks=3)
        action = {"type": ActionType.EXPLORE, "embedding": [0.5] * 64}
        context = {}

        perturbed = validator._fgsm_attack(
            action, context, validator.default_attack_config
        )

        # Should still generate perturbations (though they may be minimal)
        assert isinstance(perturbed, list)

    def test_large_embedding(self, adversarial_validator, sample_context):
        """Test handling of large embeddings."""
        action = {
            "type": ActionType.EXPLORE,
            "embedding": [0.5] * 10000,  # Very large embedding
        }

        config = AttackConfig(epsilon=0.1, num_iterations=5)

        # Should complete without hanging
        start_time = time.time()
        perturbed = adversarial_validator._pgd_attack(action, sample_context, config)
        elapsed = time.time() - start_time

        assert elapsed < 10.0  # Should not take too long
        assert isinstance(perturbed, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
