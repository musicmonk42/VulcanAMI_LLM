"""
observation_processor.py - Processes raw observations for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Contains:
- ObservationProcessor: Processes, validates, and extracts patterns from observations.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .observation_types import Observation

logger = logging.getLogger(__name__)


class ObservationProcessor:
    """Processes raw observations for the world model"""

    def __init__(self, safety_validator=None):
        self.variable_types = {}
        self.observation_history = deque(maxlen=1000)
        self.validation_rules = {}
        self.safety_validator = safety_validator

    def extract_variables(self, observation: Observation) -> Dict[str, Any]:
        """Extract and type variables from observation"""
        variables = {}

        for key, value in observation.variables.items():
            # Infer type if not known
            if key not in self.variable_types:
                self.variable_types[key] = self._infer_type(value)

            # Convert to standard format
            variables[key] = self._standardize_value(value, self.variable_types[key])

        return variables

    def detect_intervention_data(
        self, observation: Observation
    ) -> Optional[Dict[str, Any]]:
        """Detect if observation contains intervention data"""
        if observation.intervention_data:
            # Validate intervention data with safety validator
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, "validate_intervention_data"):
                        validation_result = (
                            self.safety_validator.validate_intervention_data(
                                observation.intervention_data
                            )
                        )
                        if not validation_result.get("safe", True):
                            logger.warning(
                                "Unsafe intervention data detected: %s",
                                validation_result.get("reason", "unknown"),
                            )
                            return None
                except Exception as e:
                    logger.error(
                        "Safety validator error in validate_intervention_data: %s", e
                    )
            return observation.intervention_data

        # Check for intervention markers in metadata
        if observation.metadata.get("is_intervention", False):
            intervention_data = {
                "intervened_variable": observation.metadata.get("intervened_var"),
                "intervention_value": observation.metadata.get("intervention_val"),
                "control_group": observation.metadata.get("control", None),
            }

            # Validate with safety validator
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, "validate_intervention_data"):
                        validation_result = (
                            self.safety_validator.validate_intervention_data(
                                intervention_data
                            )
                        )
                        if not validation_result.get("safe", True):
                            logger.warning(
                                "Unsafe intervention data in metadata: %s",
                                validation_result.get("reason", "unknown"),
                            )
                            return None
                except Exception as e:
                    logger.error(
                        "Safety validator error in validate_intervention_data: %s", e
                    )

            return intervention_data

        # Check for statistical anomalies that suggest intervention
        if self._detect_statistical_intervention(observation):
            return {"type": "detected", "confidence": 0.7}

        return None

    def extract_temporal_patterns(self, observation: Observation) -> Dict[str, Any]:
        """Extract temporal patterns from observation sequence"""
        self.observation_history.append(observation)

        if len(self.observation_history) < 10:
            return {}

        patterns = {"trends": {}, "cycles": {}, "anomalies": []}

        # Analyze each variable for temporal patterns
        for var_name in observation.variables.keys():
            history = [
                obs.variables.get(var_name)
                for obs in self.observation_history
                if var_name in obs.variables
            ]

            if len(history) > 5 and all(isinstance(v, (int, float)) for v in history):
                # Trend detection
                trend = self._detect_trend(history)
                if abs(trend) > 0.1:
                    patterns["trends"][var_name] = trend

                # Cycle detection
                cycle = self._detect_cycle(history)
                if cycle:
                    patterns["cycles"][var_name] = cycle

                # Anomaly detection
                if self._is_anomaly(history[-1], history[:-1]):
                    patterns["anomalies"].append(var_name)

        # Validate patterns with safety validator
        if self.safety_validator and patterns:
            try:
                if hasattr(self.safety_validator, "validate_pattern"):
                    pattern_validation = self.safety_validator.validate_pattern(
                        patterns
                    )
                    if not pattern_validation.get("safe", True):
                        logger.warning(
                            "Unsafe pattern detected: %s",
                            pattern_validation.get("reason", "unknown"),
                        )
                        # Filter out unsafe patterns
                        patterns = {"trends": {}, "cycles": {}, "anomalies": []}
            except Exception as e:
                logger.error("Safety validator error in validate_pattern: %s", e)

        return patterns

    def extract_linguistic_data(self, observation: Observation) -> Optional[str]:
        """Extract linguistic data (raw text) from observation if present"""

        # Check 'text' or 'utterance' in variables or metadata
        if "text" in observation.variables and isinstance(
            observation.variables["text"], str
        ):
            return observation.variables["text"]
        if "utterance" in observation.variables and isinstance(
            observation.variables["utterance"], str
        ):
            return observation.variables["utterance"]
        if observation.metadata.get("linguistic_input") and isinstance(
            observation.metadata["linguistic_input"], str
        ):
            return observation.metadata["linguistic_input"]

        return None

    def validate_observation(
        self, observation: Observation
    ) -> Tuple[bool, Optional[str]]:
        """Validate observation for consistency and quality"""
        # Check for required variables
        if not observation.variables and not self.extract_linguistic_data(observation):
            return False, "No variables or linguistic data in observation"

        # Check timestamp
        if observation.timestamp <= 0:
            return False, "Invalid timestamp"

        # Check variable constraints
        for var_name, value in observation.variables.items():
            if var_name in self.validation_rules:
                rule = self.validation_rules[var_name]
                if not rule(value):
                    return False, f"Variable {var_name} failed validation"

        # Check for NaN or infinite values
        # FIX: Only check numeric types to avoid "ufunc 'isnan' not supported" error
        for var_name, value in observation.variables.items():
            # Only check for NaN/inf on numeric types (int, float, np.number)
            if isinstance(value, (int, float)):
                try:
                    if np.isnan(value) or np.isinf(value):
                        return False, f"Variable {var_name} has invalid numeric value"
                except (TypeError, ValueError):
                    # Can't check NaN on this type - skip
                    pass
            elif isinstance(value, np.ndarray):
                # For arrays, check if it's a numeric dtype before calling isnan
                if np.issubdtype(value.dtype, np.number):
                    try:
                        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                            return False, f"Variable {var_name} has invalid numeric values in array"
                    except (TypeError, ValueError):
                        # Can't check NaN on this array type - skip
                        pass
            # For non-numeric types (strings, objects, etc.), skip NaN check

        # Safety validation
        if self.safety_validator:
            try:
                if hasattr(self.safety_validator, "analyze_observation_safety"):
                    safety_result = self.safety_validator.analyze_observation_safety(
                        observation
                    )
                    if not safety_result.get("safe", True):
                        return (
                            False,
                            f"Safety validation failed: {safety_result.get('reason', 'unknown')}",
                        )
            except Exception as e:
                logger.error(
                    "Safety validator error in analyze_observation_safety: %s", e
                )
                # Fail-safe: block on validator error
                return False, f"Safety validator error: {str(e)}"

        return True, None

    def _infer_type(self, value: Any) -> str:
        """Infer variable type from value"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "numeric"
        elif isinstance(value, str):
            return "categorical"
        elif isinstance(value, (list, tuple)):
            return "vector"
        else:
            return "unknown"

    def _standardize_value(self, value: Any, var_type: str) -> Any:
        """Standardize value based on type"""
        if var_type == "numeric":
            return float(value) if not isinstance(value, float) else value
        elif var_type == "boolean":
            return bool(value)
        elif var_type == "categorical":
            return str(value)
        else:
            return value

    def _detect_statistical_intervention(self, observation: Observation) -> bool:
        """Detect intervention through statistical analysis"""
        if len(self.observation_history) < 20:
            return False

        # Check for sudden distribution shifts
        # FIXED: Convert deque to list before slicing
        for var_name, current_value in observation.variables.items():
            if isinstance(current_value, (int, float)):
                history = [
                    obs.variables.get(var_name)
                    for obs in list(self.observation_history)[-20:-1]
                    if var_name in obs.variables
                    and isinstance(obs.variables[var_name], (int, float))
                ]

                if history:
                    mean = np.mean(history)
                    std = np.std(history)
                    if std > 0 and abs(current_value - mean) > 3 * std:
                        return True

        return False

    def _detect_trend(self, history: List[float]) -> float:
        """Detect linear trend in time series"""
        if len(history) < 3:
            return 0.0

        x = np.arange(len(history))
        y = np.array(history)

        # Simple linear regression
        a = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(a, y, rcond=None)[0]

        return m  # Return slope

    def _detect_cycle(self, history: List[float]) -> Optional[Dict[str, float]]:
        """Detect cyclical patterns"""
        if len(history) < 20:
            return None

        # Simple autocorrelation-based cycle detection
        data = np.array(history)
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / len(data)

        max_corr = 0
        best_lag = 0

        for lag in range(2, min(len(data) // 2, 50)):
            c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / len(data)
            corr = c_lag / c0 if c0 > 0 else 0

            if corr > max_corr:
                max_corr = corr
                best_lag = lag

        if max_corr > 0.5:  # Significant correlation
            return {"period": best_lag, "strength": max_corr}

        return None

    def _is_anomaly(
        self, value: float, history: List[float], z_threshold: float = 3.0
    ) -> bool:
        """Check if value is anomalous compared to history"""
        if len(history) < 5:
            return False

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return value != mean

        z_score = abs(value - mean) / std
        return z_score > z_threshold
