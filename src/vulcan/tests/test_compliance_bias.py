"""
Comprehensive test suite for ComplianceMapper and BiasDetector.
Tests regulatory compliance validation and multi-model bias detection.
"""

from vulcan.safety.compliance_bias import BiasDetector, ComplianceMapper, LRUCache
import numpy as np
from pathlib import Path
import time
import threading
import tempfile
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip(
    "torch", reason="PyTorch required for compliance_bias tests"
)


# Import from safety_types (with fallback)
try:
    from vulcan.safety.safety_types import ComplianceStandard
except ImportError:
    # Mock if not available
    from enum import Enum

    class ComplianceStandard(Enum):
        GDPR = "gdpr"
        HIPAA = "hipaa"
        ITU_F748_53 = "itu_f748_53"
        AI_ACT = "ai_act"
        CCPA = "ccpa"
        SOC2 = "soc2"
        ISO27001 = "iso27001"
        PCI_DSS = "pci_dss"
        COPPA = "coppa"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def compliance_mapper():
    """Create a ComplianceMapper instance."""
    return ComplianceMapper()


@pytest.fixture
def compliance_mapper_strict():
    """Create a ComplianceMapper with strict mode."""
    return ComplianceMapper(config={"strict_mode": True})


@pytest.fixture
def bias_detector():
    """Create a BiasDetector instance."""
    return BiasDetector()


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""
    return {
        "type": "data_processing",
        "confidence": 0.85,
        "data_fields": ["name", "email", "age"],
        "data_size_mb": 10.0,
        "purposes": ["analytics"],
        "contains_personal_data": True,
        "encrypted": True,
        "encryption_method": "AES-256",
        "access_controlled": True,
        "processing_documented": True,
        "audit_trail": True,
    }


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return {
        "necessary_fields": ["name", "email"],
        "necessary_data_size_mb": 5.0,
        "stated_purposes": ["analytics", "reporting"],
        "max_data_age_days": 365,
        "max_retention_days": 730,
        "user_consent": {
            "given": True,
            "specific": True,
            "informed": True,
            "freely_given": True,
        },
    }


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model export."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# LRU CACHE TESTS
# ============================================================================


class TestLRUCache:
    """Test LRU cache implementation."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(maxsize=10)
        assert cache.maxsize == 10
        assert len(cache.cache) == 0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(maxsize=5)
        cache.put("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = LRUCache(maxsize=5)
        assert cache.get("nonexistent") is None

    def test_maxsize_enforcement(self):
        """Test that cache enforces max size."""
        cache = LRUCache(maxsize=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        """Test LRU eviction order."""
        cache = LRUCache(maxsize=3)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access 'a' to make it most recent
        cache.get("a")

        # Add new item - should evict 'b' (least recently used)
        cache.put("d", 4)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("d") == 4

    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(maxsize=5)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert len(cache.cache) == 0


# ============================================================================
# COMPLIANCE MAPPER TESTS
# ============================================================================


class TestComplianceMapper:
    """Test ComplianceMapper class."""

    def test_initialization(self, compliance_mapper):
        """Test mapper initialization."""
        assert len(compliance_mapper.standards) > 0
        assert ComplianceStandard.GDPR in compliance_mapper.standards
        assert ComplianceStandard.HIPAA in compliance_mapper.standards

    def test_initialization_strict_mode(self, compliance_mapper_strict):
        """Test strict mode initialization."""
        assert compliance_mapper_strict.strict_mode is True

    def test_standards_initialization(self, compliance_mapper):
        """Test that all standards are properly initialized."""
        for standard in ComplianceStandard:
            assert standard in compliance_mapper.standards
            standard_info = compliance_mapper.standards[standard]
            assert "name" in standard_info
            assert "requirements" in standard_info
            assert "checks" in standard_info

    def test_check_compliance_single_standard(
        self, compliance_mapper, sample_action, sample_context
    ):
        """Test compliance check for single standard."""
        result = compliance_mapper.check_compliance(
            sample_action, sample_context, standards=[ComplianceStandard.GDPR]
        )

        assert isinstance(result, dict)
        assert "compliant" in result
        assert "compliance_score" in result
        assert "standard_results" in result
        assert ComplianceStandard.GDPR.value in result["standard_results"]

    def test_check_compliance_all_standards(
        self, compliance_mapper, sample_action, sample_context
    ):
        """Test compliance check for all standards."""
        result = compliance_mapper.check_compliance(sample_action, sample_context)

        assert isinstance(result, dict)
        assert "compliant" in result
        assert 0 <= result["compliance_score"] <= 1
        assert len(result["standard_results"]) > 0

    def test_check_compliance_caching(
        self, compliance_mapper, sample_action, sample_context
    ):
        """Test that compliance results are cached."""
        # First call
        result1 = compliance_mapper.check_compliance(
            sample_action, sample_context, standards=[ComplianceStandard.GDPR]
        )

        cache_size_1 = len(compliance_mapper.compliance_cache)

        # Second call with same inputs
        result2 = compliance_mapper.check_compliance(
            sample_action, sample_context, standards=[ComplianceStandard.GDPR]
        )

        cache_size_2 = len(compliance_mapper.compliance_cache)

        # Cache should be used
        assert cache_size_2 == cache_size_1
        assert result1["compliance_score"] == result2["compliance_score"]

    def test_cache_size_limit(self, compliance_mapper, sample_context):
        """Test that cache enforces size limit."""
        compliance_mapper.cache_max_size = 10

        # Generate many different actions
        for i in range(20):
            action = {"type": f"action_{i}", "data": i}
            compliance_mapper.check_compliance(
                action, sample_context, standards=[ComplianceStandard.GDPR]
            )

        # Cache should be limited
        assert (
            len(compliance_mapper.compliance_cache) <= compliance_mapper.cache_max_size
        )

    def test_clear_cache(self, compliance_mapper, sample_action, sample_context):
        """Test cache clearing."""
        compliance_mapper.check_compliance(sample_action, sample_context)
        assert len(compliance_mapper.compliance_cache) > 0

        compliance_mapper.clear_cache()
        assert len(compliance_mapper.compliance_cache) == 0

    # ========================================================================
    # GDPR Tests
    # ========================================================================

    def test_gdpr_data_minimization_pass(self, compliance_mapper):
        """Test GDPR data minimization - passing case."""
        action = {"data_fields": ["name", "email"], "data_size_mb": 5.0}
        context = {
            "necessary_fields": ["name", "email", "phone"],
            "necessary_data_size_mb": 10.0,
        }

        passed, details = compliance_mapper._check_gdpr_data_minimization(
            action, context
        )
        assert passed is True

    def test_gdpr_data_minimization_fail(self, compliance_mapper):
        """Test GDPR data minimization - failing case."""
        action = {
            "data_fields": ["name", "email", "ssn", "credit_card"],
            "data_size_mb": 50.0,
        }
        context = {"necessary_fields": ["name", "email"], "necessary_data_size_mb": 5.0}

        passed, details = compliance_mapper._check_gdpr_data_minimization(
            action, context
        )
        assert passed is False
        assert "unnecessary" in details.lower()

    def test_gdpr_purpose_limitation_pass(self, compliance_mapper):
        """Test GDPR purpose limitation - passing case."""
        action = {"purposes": ["analytics"]}
        context = {"stated_purposes": ["analytics", "reporting"]}

        passed, details = compliance_mapper._check_gdpr_purpose_limitation(
            action, context
        )
        assert passed is True

    def test_gdpr_purpose_limitation_fail(self, compliance_mapper):
        """Test GDPR purpose limitation - failing case."""
        action = {"purposes": ["analytics", "marketing", "profiling"]}
        context = {"stated_purposes": ["analytics"]}

        passed, details = compliance_mapper._check_gdpr_purpose_limitation(
            action, context
        )
        assert passed is False

    def test_gdpr_consent_pass(self, compliance_mapper):
        """Test GDPR consent - passing case."""
        action = {"requires_consent": True}
        context = {
            "user_consent": {
                "given": True,
                "withdrawn": False,
                "specific": True,
                "informed": True,
                "forced": False,
            }
        }

        passed, details = compliance_mapper._check_gdpr_consent(action, context)
        assert passed is True

    def test_gdpr_consent_fail_not_given(self, compliance_mapper):
        """Test GDPR consent - failing case (no consent)."""
        action = {"requires_consent": True}
        context = {"user_consent": {"given": False}}

        passed, details = compliance_mapper._check_gdpr_consent(action, context)
        assert passed is False
        assert "consent" in details.lower()

    def test_gdpr_consent_fail_withdrawn(self, compliance_mapper):
        """Test GDPR consent - failing case (withdrawn)."""
        action = {"requires_consent": True}
        context = {"user_consent": {"given": True, "withdrawn": True}}

        passed, details = compliance_mapper._check_gdpr_consent(action, context)
        assert passed is False
        assert "withdrawn" in details.lower()

    def test_gdpr_erasure_pass(self, compliance_mapper):
        """Test GDPR right to erasure - passing case."""
        action = {"erasure_capable": True, "stores_data": False}
        context = {"erasure_requested": False}

        passed, details = compliance_mapper._check_gdpr_erasure(action, context)
        assert passed is True

    def test_gdpr_erasure_fail(self, compliance_mapper):
        """Test GDPR right to erasure - failing case."""
        action = {"stores_data": True, "processes_data": True}
        context = {"erasure_requested": True, "legal_requirement": False}

        passed, details = compliance_mapper._check_gdpr_erasure(action, context)
        assert passed is False

    # ========================================================================
    # HIPAA Tests
    # ========================================================================

    def test_hipaa_phi_protection_pass(self, compliance_mapper):
        """Test HIPAA PHI protection - passing case."""
        action = {"contains_phi": True, "phi_encrypted": True}
        context = {"hipaa_compliant_storage": True}

        passed, details = compliance_mapper._check_hipaa_phi(action, context)
        assert passed is True

    def test_hipaa_phi_protection_fail_not_encrypted(self, compliance_mapper):
        """Test HIPAA PHI protection - failing case (not encrypted)."""
        action = {"contains_phi": True, "phi_encrypted": False}
        context = {"hipaa_compliant_storage": True}

        passed, details = compliance_mapper._check_hipaa_phi(action, context)
        assert passed is False
        assert "encrypt" in details.lower()

    def test_hipaa_access_control_pass(self, compliance_mapper):
        """Test HIPAA access control - passing case."""
        action = {"accesses_phi": True, "allowed_roles": ["doctor", "nurse"]}
        context = {
            "user_authenticated": True,
            "user_authorized": True,
            "user_role": "doctor",
        }

        passed, details = compliance_mapper._check_hipaa_access(action, context)
        assert passed is True

    def test_hipaa_access_control_fail_unauthorized(self, compliance_mapper):
        """Test HIPAA access control - failing case (unauthorized role)."""
        action = {"accesses_phi": True, "allowed_roles": ["doctor", "nurse"]}
        context = {
            "user_authenticated": True,
            "user_authorized": True,
            "user_role": "janitor",
        }

        passed, details = compliance_mapper._check_hipaa_access(action, context)
        assert passed is False
        assert "role" in details.lower()

    def test_hipaa_audit_logging_pass(self, compliance_mapper):
        """Test HIPAA audit logging - passing case."""
        action = {
            "accesses_phi": True,
            "audit_logged": True,
            "audit_retention_days": 2200,
        }
        context = {}

        passed, details = compliance_mapper._check_hipaa_audit(action, context)
        assert passed is True

    def test_hipaa_audit_logging_fail_short_retention(self, compliance_mapper):
        """Test HIPAA audit logging - failing case (short retention)."""
        action = {
            "accesses_phi": True,
            "audit_logged": True,
            "audit_retention_days": 365,
        }
        context = {}

        passed, details = compliance_mapper._check_hipaa_audit(action, context)
        assert passed is False
        assert "retention" in details.lower()

    # ========================================================================
    # ITU F.748.53 Tests
    # ========================================================================

    def test_itu_transparency_pass(self, compliance_mapper):
        """Test ITU transparency - passing case."""
        action = {
            "explanation": "This is a detailed explanation of the action.",
            "decision_traceable": True,
        }
        context = {}

        passed, details = compliance_mapper._check_itu_transparency(action, context)
        assert passed is True

    def test_itu_transparency_fail_no_explanation(self, compliance_mapper):
        """Test ITU transparency - failing case (no explanation)."""
        action = {"explanation": "", "decision_traceable": True}
        context = {}

        passed, details = compliance_mapper._check_itu_transparency(action, context)
        assert passed is False

    def test_itu_human_oversight_pass(self, compliance_mapper):
        """Test ITU human oversight - passing case."""
        action = {
            "autonomous_decision": True,
            "human_reviewable": True,
            "human_override": True,
            "high_risk": False,
        }
        context = {}

        passed, details = compliance_mapper._check_itu_human_oversight(action, context)
        assert passed is True

    def test_itu_human_oversight_fail_high_risk(self, compliance_mapper):
        """Test ITU human oversight - failing case (high-risk without monitoring)."""
        action = {
            "autonomous_decision": True,
            "human_reviewable": True,
            "human_override": True,
            "high_risk": True,
        }
        context = {"human_monitoring": False}

        passed, details = compliance_mapper._check_itu_human_oversight(action, context)
        assert passed is False
        assert "oversight" in details.lower()

    # ========================================================================
    # AI Act Tests
    # ========================================================================

    def test_ai_act_risk_assessment_pass(self, compliance_mapper):
        """Test AI Act risk assessment - passing case."""
        action = {"ai_risk_category": "limited", "risk_assessment_documented": True}
        context = {}

        passed, details = compliance_mapper._check_ai_act_risk(action, context)
        assert passed is True

    def test_ai_act_risk_assessment_fail_unacceptable(self, compliance_mapper):
        """Test AI Act risk assessment - failing case (unacceptable risk)."""
        action = {
            "ai_risk_category": "unacceptable",
            "risk_assessment_documented": True,
        }
        context = {}

        passed, details = compliance_mapper._check_ai_act_risk(action, context)
        assert passed is False
        assert "unacceptable" in details.lower()

    def test_ai_act_bias_prevention_pass(self, compliance_mapper):
        """Test AI Act bias prevention - passing case."""
        action = {
            "bias_scores": {"gender": 0.1, "race": 0.15},
            "bias_tested": True,
            "bias_mitigation": True,
        }
        context = {"max_bias_score": 0.2}

        passed, details = compliance_mapper._check_ai_act_bias(action, context)
        assert passed is True

    def test_ai_act_bias_prevention_fail(self, compliance_mapper):
        """Test AI Act bias prevention - failing case (high bias)."""
        action = {
            "bias_scores": {"gender": 0.3, "race": 0.15},
            "bias_tested": True,
            "bias_mitigation": True,
        }
        context = {"max_bias_score": 0.2}

        passed, details = compliance_mapper._check_ai_act_bias(action, context)
        assert passed is False
        assert "bias" in details.lower()

    # ========================================================================
    # Statistics and Metrics Tests
    # ========================================================================

    def test_get_compliance_stats(
        self, compliance_mapper, sample_action, sample_context
    ):
        """Test getting compliance statistics."""
        # Perform some checks
        for _ in range(5):
            compliance_mapper.check_compliance(
                sample_action, sample_context, standards=[ComplianceStandard.GDPR]
            )

        stats = compliance_mapper.get_compliance_stats()

        assert "total_checks" in stats
        assert "standards" in stats
        assert stats["total_checks"] > 0

    def test_thread_safety(self, compliance_mapper, sample_action, sample_context):
        """Test thread-safe operations."""
        results = []

        def check_compliance():
            result = compliance_mapper.check_compliance(
                sample_action, sample_context, standards=[ComplianceStandard.GDPR]
            )
            results.append(result)

        threads = [threading.Thread(target=check_compliance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)


# ============================================================================
# BIAS DETECTOR TESTS
# ============================================================================


class TestBiasDetector:
    """Test BiasDetector class."""

    def test_initialization(self, bias_detector):
        """Test bias detector initialization."""
        assert len(bias_detector.bias_models) > 0
        assert "demographic" in bias_detector.bias_models
        assert "representation" in bias_detector.bias_models
        assert "fairness" in bias_detector.bias_models

    def test_device_setup(self, bias_detector):
        """Test that models are on correct device."""
        assert bias_detector.device.type in ["cpu", "cuda"]

        for model in bias_detector.bias_models.values():
            # Check that model parameters are on the same device
            param_devices = {p.device for p in model.parameters()}
            assert len(param_devices) == 1  # All params on same device

    def test_models_in_eval_mode(self, bias_detector):
        """Test that models are in evaluation mode."""
        for model in bias_detector.bias_models.values():
            assert not model.training

    def test_detect_bias_basic(self, bias_detector):
        """Test basic bias detection."""
        action = {
            "type": "decision",
            "confidence": 0.85,
            "predicted_outcomes": {"group_a": 0.8, "group_b": 0.6},
        }
        context = {"user_diversity_score": 0.7, "historical_bias_score": 0.1}

        result = bias_detector.detect_bias(action, context)

        assert isinstance(result, dict)
        assert "bias_detected" in result
        assert "bias_scores" in result
        assert "recommendations" in result
        assert "detection_confidence" in result

    def test_detect_bias_consensus(self, bias_detector):
        """Test multi-model consensus."""
        action = {"type": "test", "confidence": 0.8}
        context = {}

        result = bias_detector.detect_bias(action, context, use_consensus=True)

        assert "consensus" in result["bias_scores"]
        assert isinstance(result["bias_scores"]["consensus"], float)
        assert 0 <= result["bias_scores"]["consensus"] <= 1

    def test_detect_bias_no_consensus(self, bias_detector):
        """Test without consensus."""
        action = {"type": "test", "confidence": 0.8}
        context = {}

        result = bias_detector.detect_bias(action, context, use_consensus=False)

        assert "bias_scores" in result
        # Consensus may or may not be present

    def test_detect_bias_caching(self, bias_detector):
        """Test that bias detection results are cached."""
        action = {"type": "test", "confidence": 0.8}
        context = {}

        # First call
        result1 = bias_detector.detect_bias(action, context)

        # Second call with same inputs (should use cache)
        result2 = bias_detector.detect_bias(action, context)

        # Results should be identical
        assert result1["bias_scores"] == result2["bias_scores"]

    def test_feature_extraction(self, bias_detector):
        """Test feature extraction."""
        action = {
            "type": "decision",
            "confidence": 0.85,
            "uncertainty": 0.1,
            "risk_score": 0.2,
        }
        context = {"user_diversity_score": 0.7, "historical_bias_score": 0.15}

        features = bias_detector._extract_features(action, context)

        assert isinstance(features, np.ndarray)
        assert features.shape == (128,)
        assert features.dtype == np.float32 or features.dtype == np.float64

    def test_extended_feature_extraction(self, bias_detector):
        """Test extended feature extraction."""
        action = {"type": "test", "confidence": 0.8}
        context = {}

        features = bias_detector._extract_extended_features(action, context)

        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)

    def test_demographic_bias_calculation(self, bias_detector):
        """Test demographic bias calculation."""
        # Create mock output
        output = torch.rand(1, 10)  # 10 demographic categories
        output = torch.softmax(output, dim=-1)

        bias_score = bias_detector._calculate_demographic_bias(output)

        assert isinstance(bias_score, float)
        assert 0 <= bias_score <= 1

    def test_demographic_bias_analysis(self, bias_detector):
        """Test detailed demographic bias analysis."""
        output = torch.rand(1, 10)
        output = torch.softmax(output, dim=-1)

        analysis = bias_detector._analyze_demographic_bias(output)

        assert "distribution" in analysis
        assert "max_category" in analysis
        assert "entropy" in analysis
        assert "uniformity" in analysis

    def test_consensus_computation(self, bias_detector):
        """Test consensus bias score computation."""
        bias_scores = {
            "demographic": 0.3,
            "representation": 0.2,
            "outcome": 0.25,
            "allocation": 0.15,
            "fairness": 0.1,
        }

        consensus = bias_detector._compute_consensus(bias_scores)

        assert isinstance(consensus, float)
        assert 0 <= consensus <= 1

    def test_bias_mitigation_recommendations(self, bias_detector):
        """Test bias mitigation recommendation generation."""
        bias_scores = {"demographic": 0.4, "representation": 0.3, "outcome": 0.2}
        detailed_analysis = {}

        recommendations = bias_detector._generate_bias_mitigations(
            bias_scores, detailed_analysis
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(r, str) for r in recommendations)

    def test_detection_confidence_calculation(self, bias_detector):
        """Test detection confidence calculation."""
        bias_scores = {"demographic": 0.3, "representation": 0.28, "outcome": 0.32}

        confidence = bias_detector._calculate_detection_confidence(bias_scores)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_high_bias_detection(self, bias_detector):
        """Test detection of high bias."""
        action = {
            "type": "decision",
            "predicted_outcomes": {"group_a": 0.9, "group_b": 0.3},
            "affects_protected_group": True,
            "differential_impact": True,
        }
        context = {"historical_bias_score": 0.5, "sensitive_attributes_present": True}

        result = bias_detector.detect_bias(action, context)

        # Should detect some bias
        assert isinstance(result["bias_detected"], bool)
        assert len(result["recommendations"]) > 0

    def test_low_bias_detection(self, bias_detector):
        """Test detection with low bias."""
        action = {"type": "neutral_action", "confidence": 0.8}
        context = {}

        result = bias_detector.detect_bias(action, context)

        assert isinstance(result["bias_detected"], bool)
        # Low bias should have lower scores
        for score in result["bias_scores"].values():
            assert score <= 1.0

    def test_text_feature_extraction(self, bias_detector):
        """Test text feature extraction."""
        text = "This is a typical example of normal behavior that always works."

        features = bias_detector._extract_text_features(text)

        assert isinstance(features, np.ndarray)
        assert features.shape == (20,)
        # Should detect bias keywords like "always", "typical", "normal"
        assert np.sum(features > 0) > 0

    def test_resource_distribution_encoding(self, bias_detector):
        """Test resource distribution encoding."""
        resources = {"cpu": 0.7, "memory": 0.5, "network": 0.3}

        features = bias_detector._encode_resource_distribution(resources)

        assert isinstance(features, np.ndarray)
        assert features.shape == (5,)

    def test_outcome_distribution_encoding(self, bias_detector):
        """Test outcome distribution encoding."""
        outcomes = {"group_a": {"probability": 0.8}, "group_b": {"probability": 0.6}}

        features = bias_detector._encode_outcome_distribution(outcomes)

        assert isinstance(features, np.ndarray)
        assert features.shape == (5,)

    def test_statistical_parity_features(self, bias_detector):
        """Test statistical parity feature calculation."""
        action = {"selection_rate": 0.7}
        context = {
            "group_rates": {"group_a": 0.8, "group_b": 0.6},
            "true_positive_rates": {"group_a": 0.9, "group_b": 0.85},
            "false_positive_rates": {"group_a": 0.1, "group_b": 0.15},
        }

        features = bias_detector._calculate_statistical_parity_features(action, context)

        assert isinstance(features, np.ndarray)
        assert features.shape == (10,)

    def test_get_bias_stats(self, bias_detector):
        """Test getting bias statistics."""
        # Perform some detections
        action = {"type": "test", "confidence": 0.8}
        context = {}

        for _ in range(5):
            bias_detector.detect_bias(action, context)

        stats = bias_detector.get_bias_stats()

        assert "total_checks" in stats
        assert "model_version" in stats
        assert stats["total_checks"] > 0

    def test_model_export_load(self, bias_detector, temp_model_dir):
        """Test model export and loading."""
        model_path = str(temp_model_dir / "bias_models.pt")

        # Export models
        bias_detector.export_models(model_path)

        assert Path(model_path).exists()

        # Create new detector and load models
        new_detector = BiasDetector()
        new_detector.load_models(model_path)

        assert new_detector.model_version == bias_detector.model_version

    def test_update_model(self, bias_detector):
        """Test model update with feedback."""
        feedback_data = [
            {"action": {"type": "test"}, "true_bias": 0.3, "detected_bias": 0.25}
        ]

        # Should not crash
        bias_detector.update_model(feedback_data)

        assert len(bias_detector.training_data) > 0

    def test_thread_safety(self, bias_detector):
        """Test thread-safe operations."""
        action = {"type": "test", "confidence": 0.8}
        context = {}

        results = []

        def detect():
            result = bias_detector.detect_bias(action, context)
            results.append(result)

        threads = [threading.Thread(target=detect) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_model_forward_pass(self, bias_detector):
        """Test that models can perform forward passes."""
        features = torch.randn(1, 128)

        with torch.no_grad():
            for model_name, model in bias_detector.bias_models.items():
                if model_name in [
                    "demographic",
                    "representation",
                    "outcome",
                    "allocation",
                ]:
                    output = model(features)
                    assert output is not None
                    assert output.shape[0] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for compliance and bias detection."""

    def test_compliance_and_bias_together(self, compliance_mapper, bias_detector):
        """Test using compliance and bias detection together."""
        action = {
            "type": "ai_decision",
            "confidence": 0.85,
            "ai_risk_category": "high",
            "bias_tested": True,
            "bias_mitigation": True,
            "bias_scores": {"gender": 0.1, "race": 0.12},
            "predicted_outcomes": {"group_a": 0.8, "group_b": 0.75},
        }
        context = {
            "risk_assessment_documented": True,
            "conformity_assessment_completed": True,
            "max_bias_score": 0.2,
        }

        # Check compliance
        compliance_result = compliance_mapper.check_compliance(
            action, context, standards=[ComplianceStandard.AI_ACT]
        )

        # Check bias
        bias_result = bias_detector.detect_bias(action, context)

        assert isinstance(compliance_result, dict)
        assert isinstance(bias_result, dict)

        # Both should provide useful information
        assert "compliant" in compliance_result
        assert "bias_detected" in bias_result

    def test_full_safety_pipeline(self, compliance_mapper, bias_detector):
        """Test complete safety validation pipeline."""
        action = {
            "type": "data_processing",
            "contains_personal_data": True,
            "encrypted": True,
            "encryption_method": "AES-256",
            "confidence": 0.85,
            "predicted_outcomes": {"group_a": 0.8, "group_b": 0.78},
        }
        context = {
            "necessary_fields": ["name", "email"],
            "stated_purposes": ["analytics"],
            "user_consent": {"given": True, "specific": True, "informed": True},
        }

        # Step 1: Compliance check
        compliance_result = compliance_mapper.check_compliance(
            action,
            context,
            standards=[ComplianceStandard.GDPR, ComplianceStandard.AI_ACT],
        )

        # Step 2: Bias detection
        bias_result = bias_detector.detect_bias(action, context)

        # Analyze combined results
        overall_safe = (
            compliance_result["compliant"] and not bias_result["bias_detected"]
        )

        assert isinstance(overall_safe, bool)

        # Generate combined recommendations
        recommendations = []
        if not compliance_result["compliant"]:
            recommendations.append("Address compliance violations")
        if bias_result["bias_detected"]:
            recommendations.extend(bias_result["recommendations"])

        assert isinstance(recommendations, list)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_action(self, compliance_mapper, bias_detector):
        """Test handling of empty action."""
        action = {}
        context = {}

        # Compliance check should handle gracefully
        compliance_result = compliance_mapper.check_compliance(action, context)
        assert isinstance(compliance_result, dict)

        # Bias detection should handle gracefully
        bias_result = bias_detector.detect_bias(action, context)
        assert isinstance(bias_result, dict)

    def test_none_values(self, compliance_mapper, bias_detector):
        """Test handling of None values."""
        action = {"type": None, "confidence": None}
        context = {"user_consent": None}

        # Should not crash
        compliance_result = compliance_mapper.check_compliance(action, context)
        assert isinstance(compliance_result, dict)

        bias_result = bias_detector.detect_bias(action, context)
        assert isinstance(bias_result, dict)

    def test_invalid_standard(self, compliance_mapper):
        """Test handling of invalid standard."""
        action = {"type": "test"}
        context = {}

        # Should skip invalid standards gracefully
        class FakeStandard:
            value = "fake_standard"

        result = compliance_mapper.check_compliance(
            action, context, standards=[FakeStandard()]
        )

        assert isinstance(result, dict)

    def test_missing_features(self, bias_detector):
        """Test bias detection with missing features."""
        action = {"type": "minimal"}
        context = {}

        # Should still work with minimal information
        result = bias_detector.detect_bias(action, context)

        assert isinstance(result, dict)
        assert "bias_scores" in result

    def test_extreme_bias_scores(self, bias_detector):
        """Test handling of extreme bias scores."""
        bias_scores = {"demographic": 0.99, "representation": 0.0, "outcome": 0.5}

        consensus = bias_detector._compute_consensus(bias_scores)

        assert 0 <= consensus <= 1

    def test_cache_expiry(self, compliance_mapper, sample_action, sample_context):
        """Test cache TTL expiry."""
        compliance_mapper.cache_ttl = 0.1  # 100ms

        # First check
        result1 = compliance_mapper.check_compliance(sample_action, sample_context)

        # Wait for cache to expire
        time.sleep(0.2)

        # Second check (cache should be expired)
        result2 = compliance_mapper.check_compliance(sample_action, sample_context)

        # Both should succeed
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
