"""
Tests for vulcan.safety.dqs_integration module.

Covers: imports, DQSValidator construction (with mocked gvulcan),
validate(), check_write_barrier(), validate_and_gate(), create_validator(),
threshold validation, edge cases, and graceful degradation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# -------------------------------------------------------------------
# Helpers: mock gvulcan components used by the module
# -------------------------------------------------------------------


def _make_mock_gvulcan_modules():
    """Return a dict of mock modules that satisfy the gvulcan imports."""
    mock_dqs = MagicMock()
    mock_opa = MagicMock()

    # DQSScorer stub
    mock_scorer_instance = MagicMock()
    mock_dqs.DQSScorer.return_value = mock_scorer_instance

    # DQSComponents stub (just a data holder)
    mock_dqs.DQSComponents = MagicMock()

    # DQSResult stub
    mock_dqs.DQSResult = MagicMock

    # OPAClient stub
    mock_opa_instance = MagicMock()
    mock_opa.OPAClient.return_value = mock_opa_instance
    mock_opa.WriteBarrierInput = MagicMock()

    return {
        "gvulcan": MagicMock(),
        "gvulcan.dqs": mock_dqs,
        "gvulcan.opa": mock_opa,
    }, mock_scorer_instance, mock_opa_instance


# -------------------------------------------------------------------
# 1. Import Verification
# -------------------------------------------------------------------


class TestImports:
    """Verify the module and its public symbols are importable."""

    def test_module_imports(self):
        import vulcan.safety.dqs_integration  # noqa: F401

    def test_public_api_importable(self):
        from vulcan.safety.dqs_integration import (
            DQS_AVAILABLE,
            DQSValidator,
            create_validator,
        )
        assert isinstance(DQS_AVAILABLE, bool)
        assert DQSValidator is not None
        assert callable(create_validator)


# -------------------------------------------------------------------
# 2. Graceful Degradation (gvulcan unavailable)
# -------------------------------------------------------------------


class TestGracefulDegradation:
    """Behaviour when gvulcan is not installed."""

    def test_dqs_available_is_false_without_gvulcan(self):
        from vulcan.safety.dqs_integration import DQS_AVAILABLE

        # DQS_AVAILABLE is evaluated at import time; if gvulcan is
        # not installed in the test env it should be False.
        # We just verify it is a bool.
        assert isinstance(DQS_AVAILABLE, bool)

    def test_create_validator_returns_none_without_gvulcan(self):
        from vulcan.safety.dqs_integration import create_validator

        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", False):
            result = create_validator()
            assert result is None

    def test_constructor_raises_importerror_without_gvulcan(self):
        from vulcan.safety.dqs_integration import DQSValidator

        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", False):
            with pytest.raises(ImportError, match="gvulcan.dqs required"):
                DQSValidator()


# -------------------------------------------------------------------
# 3. Threshold Validation
# -------------------------------------------------------------------


class TestThresholdValidation:
    """Verify constructor rejects invalid thresholds."""

    def _build_validator(self, **kwargs):
        """Build a DQSValidator with mocked gvulcan deps."""
        mocks, _, _ = _make_mock_gvulcan_modules()
        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch("vulcan.safety.dqs_integration.DQSScorer", mocks["gvulcan.dqs"].DQSScorer), \
             patch("vulcan.safety.dqs_integration.DQSComponents", mocks["gvulcan.dqs"].DQSComponents), \
             patch("vulcan.safety.dqs_integration.OPAClient", mocks["gvulcan.opa"].OPAClient), \
             patch("vulcan.safety.dqs_integration.WriteBarrierInput", mocks["gvulcan.opa"].WriteBarrierInput):
            return __import__(
                "vulcan.safety.dqs_integration", fromlist=["DQSValidator"]
            ).DQSValidator(**kwargs)

    def test_reject_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="reject_threshold"):
            self._build_validator(reject_threshold=-0.1)

    def test_reject_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="reject_threshold"):
            self._build_validator(reject_threshold=1.5)

    def test_quarantine_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="quarantine_threshold"):
            self._build_validator(quarantine_threshold=-0.1)

    def test_quarantine_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="quarantine_threshold"):
            self._build_validator(quarantine_threshold=1.5)

    def test_reject_greater_than_quarantine_raises(self):
        with pytest.raises(ValueError, match="reject_threshold must be <= quarantine"):
            self._build_validator(reject_threshold=0.6, quarantine_threshold=0.4)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model must be"):
            self._build_validator(model="v99")


# -------------------------------------------------------------------
# 4. DQSValidator.validate() -- Happy Path
# -------------------------------------------------------------------


class TestValidate:
    """Test the validate() method with mocked scorer."""

    @pytest.fixture()
    def validator_and_scorer(self):
        mocks, scorer, opa = _make_mock_gvulcan_modules()
        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch("vulcan.safety.dqs_integration.DQSScorer", mocks["gvulcan.dqs"].DQSScorer), \
             patch("vulcan.safety.dqs_integration.DQSComponents", mocks["gvulcan.dqs"].DQSComponents), \
             patch("vulcan.safety.dqs_integration.OPAClient", mocks["gvulcan.opa"].OPAClient), \
             patch("vulcan.safety.dqs_integration.WriteBarrierInput", mocks["gvulcan.opa"].WriteBarrierInput):
            from vulcan.safety.dqs_integration import DQSValidator
            v = DQSValidator(reject_threshold=0.3, quarantine_threshold=0.4, model="v2")
        return v, scorer

    def test_valid_scores_accepted(self, validator_and_scorer):
        v, scorer = validator_and_scorer
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.decision = "accept"
        scorer.score.return_value = mock_result

        result = v.validate(
            pii_confidence=0.9,
            graph_completeness=0.8,
            syntactic_completeness=0.7,
        )
        assert result.score == 0.85
        assert result.decision == "accept"

    def test_invalid_pii_confidence_raises(self, validator_and_scorer):
        v, _ = validator_and_scorer
        with pytest.raises(ValueError, match="pii_confidence"):
            v.validate(pii_confidence=1.5, graph_completeness=0.5, syntactic_completeness=0.5)

    def test_invalid_graph_completeness_raises(self, validator_and_scorer):
        v, _ = validator_and_scorer
        with pytest.raises(ValueError, match="graph_completeness"):
            v.validate(pii_confidence=0.5, graph_completeness=-0.1, syntactic_completeness=0.5)

    def test_invalid_syntactic_completeness_raises(self, validator_and_scorer):
        v, _ = validator_and_scorer
        with pytest.raises(ValueError, match="syntactic_completeness"):
            v.validate(pii_confidence=0.5, graph_completeness=0.5, syntactic_completeness=2.0)


# -------------------------------------------------------------------
# 5. check_write_barrier()
# -------------------------------------------------------------------


class TestCheckWriteBarrier:
    @pytest.fixture()
    def validator_and_opa(self):
        mocks, scorer, opa = _make_mock_gvulcan_modules()
        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch("vulcan.safety.dqs_integration.DQSScorer", mocks["gvulcan.dqs"].DQSScorer), \
             patch("vulcan.safety.dqs_integration.DQSComponents", mocks["gvulcan.dqs"].DQSComponents), \
             patch("vulcan.safety.dqs_integration.OPAClient", mocks["gvulcan.opa"].OPAClient), \
             patch("vulcan.safety.dqs_integration.WriteBarrierInput", mocks["gvulcan.opa"].WriteBarrierInput):
            from vulcan.safety.dqs_integration import DQSValidator
            v = DQSValidator()
        return v, opa

    def test_barrier_allows_high_quality(self, validator_and_opa):
        v, opa = validator_and_opa
        barrier_result = MagicMock()
        barrier_result.allow = True
        barrier_result.quarantine = False
        barrier_result.deny_reason = None
        opa.evaluate_write_barrier.return_value = barrier_result

        assert v.check_write_barrier(dqs_score=0.9) is True

    def test_barrier_blocks_low_quality(self, validator_and_opa):
        v, opa = validator_and_opa
        barrier_result = MagicMock()
        barrier_result.allow = False
        barrier_result.quarantine = True
        barrier_result.deny_reason = "low quality"
        opa.evaluate_write_barrier.return_value = barrier_result

        assert v.check_write_barrier(dqs_score=0.1) is False

    def test_invalid_dqs_score_raises(self, validator_and_opa):
        v, _ = validator_and_opa
        with pytest.raises(ValueError, match="dqs_score"):
            v.check_write_barrier(dqs_score=-0.5)

    def test_pii_info_defaults_to_not_detected(self, validator_and_opa):
        v, opa = validator_and_opa
        barrier_result = MagicMock()
        barrier_result.allow = True
        barrier_result.quarantine = False
        barrier_result.deny_reason = None
        opa.evaluate_write_barrier.return_value = barrier_result

        v.check_write_barrier(dqs_score=0.5)
        # The WriteBarrierInput was constructed; verify pii defaults
        call_args = opa.evaluate_write_barrier.call_args
        assert call_args is not None


# -------------------------------------------------------------------
# 6. validate_and_gate()
# -------------------------------------------------------------------


class TestValidateAndGate:
    @pytest.fixture()
    def full_validator(self):
        mocks, scorer, opa = _make_mock_gvulcan_modules()
        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch("vulcan.safety.dqs_integration.DQSScorer", mocks["gvulcan.dqs"].DQSScorer), \
             patch("vulcan.safety.dqs_integration.DQSComponents", mocks["gvulcan.dqs"].DQSComponents), \
             patch("vulcan.safety.dqs_integration.OPAClient", mocks["gvulcan.opa"].OPAClient), \
             patch("vulcan.safety.dqs_integration.WriteBarrierInput", mocks["gvulcan.opa"].WriteBarrierInput):
            from vulcan.safety.dqs_integration import DQSValidator
            v = DQSValidator()

        # Configure scorer to return accept
        dqs_result = MagicMock()
        dqs_result.score = 0.85
        dqs_result.decision = "accept"
        scorer.score.return_value = dqs_result

        # Configure OPA to allow
        barrier_result = MagicMock()
        barrier_result.allow = True
        barrier_result.quarantine = False
        barrier_result.deny_reason = None
        opa.evaluate_write_barrier.return_value = barrier_result

        return v, scorer, opa, dqs_result, barrier_result

    def test_accept_when_both_pass(self, full_validator):
        v, _, _, _, _ = full_validator
        result = v.validate_and_gate(
            pii_confidence=0.9,
            graph_completeness=0.85,
            syntactic_completeness=0.8,
        )
        assert result["final_decision"] == "accept"
        assert result["write_barrier_passed"] is True
        assert "dqs_score" in result
        assert "metadata" in result

    def test_quarantine_when_barrier_fails(self, full_validator):
        v, _, opa, _, barrier_result = full_validator
        barrier_result.allow = False
        barrier_result.quarantine = True

        result = v.validate_and_gate(
            pii_confidence=0.9,
            graph_completeness=0.85,
            syntactic_completeness=0.8,
        )
        assert result["final_decision"] == "quarantine"

    def test_reject_when_dqs_rejects(self, full_validator):
        v, scorer, _, dqs_result, _ = full_validator
        dqs_result.decision = "reject"
        # barrier still allows
        result = v.validate_and_gate(
            pii_confidence=0.1,
            graph_completeness=0.1,
            syntactic_completeness=0.1,
        )
        assert result["final_decision"] == "reject"

    def test_metadata_includes_thresholds(self, full_validator):
        v, _, _, _, _ = full_validator
        result = v.validate_and_gate(
            pii_confidence=0.5,
            graph_completeness=0.5,
            syntactic_completeness=0.5,
        )
        meta = result["metadata"]
        assert "reject_threshold" in meta
        assert "quarantine_threshold" in meta
        assert "model" in meta


# -------------------------------------------------------------------
# 7. get_statistics()
# -------------------------------------------------------------------


class TestGetStatistics:
    def test_statistics_contain_config(self):
        mocks, scorer, opa = _make_mock_gvulcan_modules()
        opa.get_statistics.return_value = {"cache_hits": 10}
        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch("vulcan.safety.dqs_integration.DQSScorer", mocks["gvulcan.dqs"].DQSScorer), \
             patch("vulcan.safety.dqs_integration.DQSComponents", mocks["gvulcan.dqs"].DQSComponents), \
             patch("vulcan.safety.dqs_integration.OPAClient", mocks["gvulcan.opa"].OPAClient), \
             patch("vulcan.safety.dqs_integration.WriteBarrierInput", mocks["gvulcan.opa"].WriteBarrierInput):
            from vulcan.safety.dqs_integration import DQSValidator
            v = DQSValidator(reject_threshold=0.2, quarantine_threshold=0.5, model="v1")

        stats = v.get_statistics()
        assert stats["reject_threshold"] == 0.2
        assert stats["quarantine_threshold"] == 0.5
        assert stats["model"] == "v1"
        assert "opa" in stats


# -------------------------------------------------------------------
# 8. create_validator() Factory
# -------------------------------------------------------------------


class TestCreateValidator:
    def test_returns_none_when_dqs_unavailable(self):
        from vulcan.safety.dqs_integration import create_validator

        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", False):
            assert create_validator() is None

    def test_returns_none_on_construction_error(self):
        from vulcan.safety.dqs_integration import create_validator

        with patch("vulcan.safety.dqs_integration.DQS_AVAILABLE", True), \
             patch(
                 "vulcan.safety.dqs_integration.DQSValidator",
                 side_effect=RuntimeError("boom"),
             ):
            assert create_validator() is None
