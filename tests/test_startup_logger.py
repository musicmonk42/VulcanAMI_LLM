"""
Test startup logger functionality for comprehensive system initialization logging.
Validates that all components are properly logged during system initialization.
"""

import pytest
import logging
from io import StringIO
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_startup_logger_import():
    """Test that startup logger can be imported."""
    from startup_logger import StartupLogger, get_startup_logger
    
    logger = get_startup_logger()
    assert logger is not None
    assert isinstance(logger, StartupLogger)


def test_startup_logger_service_mount():
    """Test logging of service mounts."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_service_mount("VULCAN", "/vulcan", "Test description", success=True)
    
    assert "VULCAN" in sl.services_mounted
    assert sl.services_mounted["VULCAN"] is True


def test_startup_logger_component_init():
    """Test logging of component initialization."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_component_init(
        "GraphixVulcanLLM",
        version="2.0.2",
        details=["512-dimensional model", "6 layers", "8 heads"],
        success=True
    )
    
    assert "GraphixVulcanLLM" in sl.components_status
    status = sl.components_status["GraphixVulcanLLM"]
    assert status["version"] == "2.0.2"
    assert status["success"] is True


def test_startup_logger_warning():
    """Test logging of warnings."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_warning("Test warning message", note="Test note")
    
    assert len(sl.warnings) > 0
    assert sl.warnings[0] == "Test warning message"


def test_graphix_vulcan_llm_logging():
    """Test GraphixVulcanLLM specific logging."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_graphix_vulcan_llm(
        version="2.0.2",
        dimensions=512,
        layers=6,
        heads=8,
        available=True
    )
    
    assert "GraphixVulcanLLM" in sl.components_status
    status = sl.components_status["GraphixVulcanLLM"]
    assert status["version"] == "2.0.2"
    assert status["success"] is True
    assert len(status["details"]) > 0
    assert any("512-dimensional" in detail for detail in status["details"])


def test_world_model_logging():
    """Test World Model logging."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    components = {
        "Causal graphs": True,
        "Prediction engine": True,
        "Dynamics model": False,
        "Correlation tracker": True
    }
    
    sl.log_world_model(components)
    
    assert "World Model" in sl.components_status
    status = sl.components_status["World Model"]
    # Should be False since not all components are available
    assert status["success"] is False


def test_safety_layer_logging():
    """Test Safety Layer logging."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    components = {
        "Neural safety validators": True,
        "Formal verification": True,
        "Compliance/bias detection": True,
        "CSIU enforcement": True
    }
    
    sl.log_safety_layer(components)
    
    assert "Safety Layer" in sl.components_status
    status = sl.components_status["Safety Layer"]
    assert status["success"] is True


def test_meta_reasoning_logging():
    """Test Meta-reasoning logging."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    objectives = [
        "Epistemic curiosity",
        "Competence improvement",
        "Social collaboration",
        "Efficiency optimization",
        "Safety preservation",
        "Value alignment"
    ]
    
    sl.log_meta_reasoning(
        objectives=objectives,
        auto_apply=True,
        approval_required=False,
        available=True
    )
    
    assert "Meta-reasoning" in sl.components_status
    status = sl.components_status["Meta-reasoning"]
    assert status["success"] is True
    assert len(status["details"]) > 0
    assert any("6 objectives" in detail for detail in status["details"])
    assert any("Auto-apply enabled: Yes" in detail for detail in status["details"])


def test_hardware_logging():
    """Test Hardware logging."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_hardware(
        backend="CPU",
        emulator_type="Analog Photonic",
        available=True
    )
    
    assert "Hardware" in sl.components_status
    status = sl.components_status["Hardware"]
    assert status["success"] is True


def test_startup_summary():
    """Test startup summary generation."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    
    # Add some test data
    sl.services_mounted["VULCAN"] = True
    sl.services_mounted["Arena"] = True
    sl.components_status["TestComponent"] = {
        "version": "1.0.0",
        "success": True,
        "details": []
    }
    sl.warnings.append("Test warning")
    
    # This should not raise an exception
    sl.log_startup_summary()


def test_full_vulcan_startup_logging():
    """Test full VULCAN startup logging."""
    from startup_logger import log_vulcan_startup
    
    # This should not raise any exceptions
    log_vulcan_startup()


def test_six_objectives_present():
    """Test that exactly 6 objectives are logged for meta-reasoning."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    objectives = [
        "Epistemic curiosity (knowledge-seeking)",
        "Competence improvement (skill acquisition)",
        "Social collaboration (multi-agent coordination)",
        "Efficiency optimization (resource utilization)",
        "Safety preservation (risk mitigation)",
        "Value alignment (human preference learning)"
    ]
    
    assert len(objectives) == 6, "Should have exactly 6 objectives"
    
    sl.log_meta_reasoning(
        objectives=objectives,
        auto_apply=True,
        approval_required=False,
        available=True
    )
    
    # Check that 6 objectives are mentioned in details
    status = sl.components_status["Meta-reasoning"]
    details_text = " ".join(status["details"])
    assert "6 objectives" in details_text


def test_model_dimensions():
    """Test that GraphixVulcanLLM has correct dimensions (512-dim, 6 layers, 8 heads)."""
    from startup_logger import StartupLogger
    
    sl = StartupLogger()
    sl.log_graphix_vulcan_llm(
        version="2.0.2",
        dimensions=512,
        layers=6,
        heads=8,
        available=True
    )
    
    status = sl.components_status["GraphixVulcanLLM"]
    details_text = " ".join(status["details"])
    
    assert "512-dimensional" in details_text
    assert "6 layers" in details_text
    assert "8 heads" in details_text


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
