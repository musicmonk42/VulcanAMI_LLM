"""
Conformal Prediction and Confidence Calibration Package

This package provides comprehensive confidence calibration and conformal prediction
capabilities for the VULCAN system. It includes multiple calibration methods and
metrics for ensuring well-calibrated confidence scores across the system.

Key Components:
- CalibratedDecisionMaker: Main calibration system integrating multiple methods
- CalibrationData/CalibrationMetrics: Data structures for calibration
- TemperatureScaling: Post-hoc calibration via temperature parameter
- IsotonicCalibration: Isotonic regression calibration
- PlattScaling: Platt scaling (logistic regression) calibration
- BetaCalibration: Beta distribution calibration
- ConformalPredictor: Conformal prediction for uncertainty quantification

Note: This is distinct from vulcan.reasoning.selection.tool_selector.ToolConfidenceCalibrator,
which is a simplified, tool-specific calibrator using only isotonic regression for the
tool selection system.

Author: VulcanAMI Team
License: Proprietary
"""

import logging

logger = logging.getLogger(__name__)

# Import all public classes and functions
try:
    from .confidence_calibration import (
        CalibratedDecisionMaker,
        CalibrationData,
        CalibrationMetrics,
        TemperatureScaling,
        IsotonicCalibration,
        PlattScaling,
        BetaCalibration,
        ConformalPredictor,
    )
    
    _CALIBRATION_AVAILABLE = True
    logger.debug("Conformal calibration components loaded successfully")
    
except ImportError as e:
    logger.warning(f"Failed to import calibration components: {e}")
    _CALIBRATION_AVAILABLE = False
    
    # Create placeholder None values
    CalibratedDecisionMaker = None
    CalibrationData = None
    CalibrationMetrics = None
    TemperatureScaling = None
    IsotonicCalibration = None
    PlattScaling = None
    BetaCalibration = None
    ConformalPredictor = None

# Import security utilities
try:
    from .security_fixes import safe_pickle_load, RestrictedUnpickler
    
    _SECURITY_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Failed to import security utilities: {e}")
    _SECURITY_AVAILABLE = False
    
    safe_pickle_load = None
    RestrictedUnpickler = None


# Package metadata
__version__ = "1.0.0"
__author__ = "VulcanAMI Team"

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Main calibration system
    "CalibratedDecisionMaker",
    
    # Data structures
    "CalibrationData",
    "CalibrationMetrics",
    
    # Calibration methods
    "TemperatureScaling",
    "IsotonicCalibration",
    "PlattScaling",
    "BetaCalibration",
    
    # Conformal prediction
    "ConformalPredictor",
    
    # Security utilities
    "safe_pickle_load",
    "RestrictedUnpickler",
    
    # Availability flags
    "_CALIBRATION_AVAILABLE",
    "_SECURITY_AVAILABLE",
]


def get_package_info():
    """
    Get information about the conformal package.
    
    Returns:
        Dictionary with package information and availability status
    """
    return {
        "version": __version__,
        "author": __author__,
        "calibration_available": _CALIBRATION_AVAILABLE,
        "security_available": _SECURITY_AVAILABLE,
        "components": {
            "calibrated_decision_maker": CalibratedDecisionMaker is not None,
            "calibration_data": CalibrationData is not None,
            "temperature_scaling": TemperatureScaling is not None,
            "isotonic_calibration": IsotonicCalibration is not None,
            "platt_scaling": PlattScaling is not None,
            "beta_calibration": BetaCalibration is not None,
            "conformal_predictor": ConformalPredictor is not None,
        },
    }


# Log package initialization
if _CALIBRATION_AVAILABLE:
    logger.info(f"Conformal package v{__version__} loaded successfully")
else:
    logger.warning(f"Conformal package v{__version__} loaded with limited functionality")
