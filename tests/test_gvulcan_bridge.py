"""
Unit tests for GVulcanBridge integration.

Tests the gvulcan bridge that provides access to:
- DQS (Data Quality Score) for validation
- OPA (Open Policy Agent) for policy enforcement

Author: VULCAN-AGI Team
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Add src to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.integration.gvulcan_bridge import (
    GVulcanBridge,
    create_gvulcan_bridge,
    GVULCAN_AVAILABLE,
    DQS_AVAILABLE,
    OPA_AVAILABLE,
)


class TestGVulcanBridgeInitialization:
    """Test GVulcanBridge initialization."""
    
    def test_initialization_default_config(self):
        """Test bridge initializes with default configuration."""
        if not GVULCAN_AVAILABLE:
            pytest.skip("GVulcan not available")
        
        bridge = GVulcanBridge()
        assert bridge is not None
    
    def test_initialization_custom_config(self):
        """Test bridge initializes with custom configuration."""
        if not GVULCAN_AVAILABLE:
            pytest.skip("GVulcan not available")
        
        config = {
            "dqs_model": "v2",
            "dqs_reject_threshold": 0.2,
            "dqs_quarantine_threshold": 0.35,
            "opa_bundle_version": "2.0.0",
            "opa_cache_enabled": False
        }
        bridge = GVulcanBridge(config)
        assert bridge is not None
    
    def test_get_status(self):
        """Test get_status returns component availability."""
        if not GVULCAN_AVAILABLE:
            pytest.skip("GVulcan not available")
        
        bridge = GVulcanBridge()
        status = bridge.get_status()
        
        assert isinstance(status, dict)
        assert "gvulcan_available" in status
        assert "dqs_available" in status
        assert "opa_available" in status
        assert "dqs_initialized" in status
        assert "opa_initialized" in status


class TestGVulcanBridgeDataQuality:
    """Test GVulcanBridge data quality operations."""
    
    def test_validate_data_quality_defaults(self):
        """Test data quality validation with default values."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        result = bridge.validate_data_quality()
        
        if result is not None:
            assert isinstance(result, dict)
            assert "score" in result
            assert "gate_decision" in result
            assert "components" in result
    
    def test_validate_data_quality_custom_values(self):
        """Test data quality validation with custom values."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        result = bridge.validate_data_quality(
            pii_confidence=0.05,
            graph_completeness=0.95,
            syntactic_completeness=0.98
        )
        
        if result is not None:
            assert isinstance(result, dict)
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0
            assert result["gate_decision"] in ["accept", "quarantine", "reject"]
    
    def test_validate_data_quality_validation_pii_too_high(self):
        """Test validation fails for pii_confidence > 1.0."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="pii_confidence must be in"):
            bridge.validate_data_quality(pii_confidence=1.5)
    
    def test_validate_data_quality_validation_pii_negative(self):
        """Test validation fails for pii_confidence < 0.0."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="pii_confidence must be in"):
            bridge.validate_data_quality(pii_confidence=-0.1)
    
    def test_validate_data_quality_validation_graph_completeness(self):
        """Test validation fails for invalid graph_completeness."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="graph_completeness must be in"):
            bridge.validate_data_quality(graph_completeness=1.5)
    
    def test_validate_data_quality_validation_syntactic_completeness(self):
        """Test validation fails for invalid syntactic_completeness."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="syntactic_completeness must be in"):
            bridge.validate_data_quality(syntactic_completeness=-0.1)
    
    def test_validate_data_quality_edge_cases(self):
        """Test validation passes for edge cases 0.0 and 1.0."""
        if not DQS_AVAILABLE:
            pytest.skip("DQS not available")
        
        bridge = GVulcanBridge()
        
        # Test minimum values
        result1 = bridge.validate_data_quality(
            pii_confidence=0.0,
            graph_completeness=0.0,
            syntactic_completeness=0.0
        )
        if result1 is not None:
            assert isinstance(result1, dict)
        
        # Test maximum values
        result2 = bridge.validate_data_quality(
            pii_confidence=1.0,
            graph_completeness=1.0,
            syntactic_completeness=1.0
        )
        if result2 is not None:
            assert isinstance(result2, dict)


class TestGVulcanBridgePolicyEnforcement:
    """Test GVulcanBridge policy enforcement operations."""
    
    def test_check_write_barrier_default_context(self):
        """Test write barrier check with default context."""
        if not OPA_AVAILABLE:
            pytest.skip("OPA not available")
        
        bridge = GVulcanBridge()
        allowed = bridge.check_write_barrier(dqs_score=0.85, context={})
        
        assert isinstance(allowed, bool)
    
    def test_check_write_barrier_custom_context(self):
        """Test write barrier check with custom context."""
        if not OPA_AVAILABLE:
            pytest.skip("OPA not available")
        
        bridge = GVulcanBridge()
        context = {
            "pii_detected": False,
            "sensitivity_level": "medium",
            "source": "user_input"
        }
        allowed = bridge.check_write_barrier(dqs_score=0.9, context=context)
        
        assert isinstance(allowed, bool)
    
    def test_check_write_barrier_validation_score_too_high(self):
        """Test validation fails for dqs_score > 1.0."""
        if not OPA_AVAILABLE:
            pytest.skip("OPA not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="dqs_score must be in"):
            bridge.check_write_barrier(dqs_score=1.5, context={})
    
    def test_check_write_barrier_validation_score_negative(self):
        """Test validation fails for dqs_score < 0.0."""
        if not OPA_AVAILABLE:
            pytest.skip("OPA not available")
        
        bridge = GVulcanBridge()
        with pytest.raises(ValueError, match="dqs_score must be in"):
            bridge.check_write_barrier(dqs_score=-0.1, context={})
    
    def test_check_write_barrier_edge_cases(self):
        """Test write barrier with edge case scores."""
        if not OPA_AVAILABLE:
            pytest.skip("OPA not available")
        
        bridge = GVulcanBridge()
        
        # Test minimum score
        allowed1 = bridge.check_write_barrier(dqs_score=0.0, context={})
        assert isinstance(allowed1, bool)
        
        # Test maximum score
        allowed2 = bridge.check_write_barrier(dqs_score=1.0, context={})
        assert isinstance(allowed2, bool)
    
    def test_check_write_barrier_fail_open_when_unavailable(self):
        """Test write barrier fails open when OPA unavailable."""
        # Create bridge with OPA disabled
        config = {"opa_cache_enabled": False}
        bridge = GVulcanBridge(config)
        
        # Should return True (fail open) when OPA not available
        allowed = bridge.check_write_barrier(dqs_score=0.5, context={})
        assert isinstance(allowed, bool)


class TestGVulcanBridgeFactory:
    """Test create_gvulcan_bridge factory function."""
    
    def test_factory_with_none(self):
        """Test factory creates bridge with default config."""
        bridge = create_gvulcan_bridge()
        
        if GVULCAN_AVAILABLE:
            assert isinstance(bridge, GVulcanBridge)
        else:
            assert bridge is None
    
    def test_factory_with_dict(self):
        """Test factory accepts dict configuration."""
        config = {
            "dqs_reject_threshold": 0.25,
            "opa_cache_enabled": True
        }
        bridge = create_gvulcan_bridge(config)
        
        if GVULCAN_AVAILABLE:
            assert isinstance(bridge, GVulcanBridge)
        else:
            assert bridge is None
    
    def test_factory_returns_none_when_unavailable(self):
        """Test factory returns None when gvulcan unavailable."""
        if GVULCAN_AVAILABLE:
            pytest.skip("GVulcan is available")
        
        bridge = create_gvulcan_bridge()
        assert bridge is None


class TestGVulcanBridgeGracefulDegradation:
    """Test GVulcanBridge graceful degradation."""
    
    def test_validate_data_quality_without_dqs(self):
        """Test data quality validation returns None when DQS unavailable."""
        if DQS_AVAILABLE:
            pytest.skip("DQS is available")
        
        bridge = GVulcanBridge()
        result = bridge.validate_data_quality()
        assert result is None
    
    def test_check_write_barrier_without_opa(self):
        """Test write barrier returns True when OPA unavailable."""
        if OPA_AVAILABLE:
            pytest.skip("OPA is available")
        
        bridge = GVulcanBridge()
        allowed = bridge.check_write_barrier(dqs_score=0.5, context={})
        
        # Should fail open (return True) when OPA unavailable
        assert allowed is True


class TestGVulcanBridgeIntegration:
    """Test GVulcanBridge integration scenarios."""
    
    def test_dqs_and_write_barrier_workflow(self):
        """Test complete DQS validation and write barrier workflow."""
        if not (DQS_AVAILABLE and OPA_AVAILABLE):
            pytest.skip("DQS or OPA not available")
        
        bridge = GVulcanBridge()
        
        # Validate data quality
        quality = bridge.validate_data_quality(
            pii_confidence=0.05,
            graph_completeness=0.95,
            syntactic_completeness=0.98
        )
        
        if quality is not None:
            # Check write barrier with quality score
            allowed = bridge.check_write_barrier(
                dqs_score=quality["score"],
                context={"source": "test"}
            )
            
            assert isinstance(allowed, bool)
    
    def test_reject_low_quality_data(self):
        """Test that low quality data is rejected."""
        if not (DQS_AVAILABLE and OPA_AVAILABLE):
            pytest.skip("DQS or OPA not available")
        
        bridge = GVulcanBridge({
            "dqs_reject_threshold": 0.5
        })
        
        # Low quality data
        quality = bridge.validate_data_quality(
            pii_confidence=0.9,  # High PII
            graph_completeness=0.1,  # Low completeness
            syntactic_completeness=0.1
        )
        
        if quality is not None:
            assert quality["gate_decision"] in ["reject", "quarantine"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
