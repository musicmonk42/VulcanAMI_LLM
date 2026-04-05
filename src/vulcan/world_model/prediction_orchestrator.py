"""
prediction_orchestrator.py - High-level prediction management for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Contains:
- PredictionManager: Manages predictions with uncertainty quantification,
  causal path validation, dynamics application, and confidence calibration.
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from .observation_types import ComponentIntegrationError, ModelContext

logger = logging.getLogger(__name__)


class PredictionManager:
    """Manages predictions with uncertainty quantification"""

    def __init__(self, world_model):
        self.world_model = world_model
        self.prediction_history = deque(maxlen=100)

    def predict(self, action: Any, context: ModelContext) -> Any:
        """Make calibrated prediction"""
        from . import world_model_core

        if not world_model_core.PREDICTION_ENGINE_AVAILABLE:
            logger.error("Cannot make prediction - PredictionEngine not available")
            # Return default prediction
            Prediction = world_model_core.Prediction
            if Prediction:
                return Prediction(
                    expected=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    confidence=0.0,
                    method="unavailable",
                )
            else:
                raise ComponentIntegrationError("Prediction system not available")

        Prediction = world_model_core.Prediction

        # EXAMINE: Find causal paths
        paths = []
        if world_model_core.CAUSAL_GRAPH_AVAILABLE:
            paths = self.world_model.causal_graph.find_all_paths(
                action, context.targets
            )

        # Validate causal paths with safety validator
        if self.world_model.safety_validator and paths:
            validated_paths = []
            for path in paths:
                # Use the new path.strengths property or get_strengths() method
                if hasattr(path, "nodes") and (
                    hasattr(path, "strengths") or hasattr(path, "get_strengths")
                ):
                    try:
                        # Try property first, then method
                        if hasattr(path, "strengths"):
                            strengths = path.strengths
                        else:
                            strengths = path.get_strengths()

                        if hasattr(
                            self.world_model.safety_validator, "validate_causal_path"
                        ):
                            path_validation = (
                                self.world_model.safety_validator.validate_causal_path(
                                    path.nodes, strengths
                                )
                            )
                            if path_validation.get("safe", True):
                                validated_paths.append(path)
                            else:
                                logger.warning(
                                    "Unsafe causal path blocked: %s",
                                    path_validation.get("reason", "unknown"),
                                )
                        else:
                            validated_paths.append(path)
                    except Exception as e:
                        logger.error("Error validating causal path: %s", e)
                        validated_paths.append(path)  # Allow on error
                elif hasattr(path, "nodes") and hasattr(path, "edges"):
                    # Fallback: extract from edges directly
                    try:
                        strengths = [edge[2] for edge in path.edges]
                        if hasattr(
                            self.world_model.safety_validator, "validate_causal_path"
                        ):
                            path_validation = (
                                self.world_model.safety_validator.validate_causal_path(
                                    path.nodes, strengths
                                )
                            )
                            if path_validation.get("safe", True):
                                validated_paths.append(path)
                            else:
                                logger.warning(
                                    "Unsafe causal path blocked: %s",
                                    path_validation.get("reason", "unknown"),
                                )
                        else:
                            validated_paths.append(path)
                    except Exception as e:
                        logger.error("Error validating causal path: %s", e)
                        validated_paths.append(path)  # Allow on error
                else:
                    validated_paths.append(path)  # Allow paths without validation info
            paths = validated_paths

        # SELECT: Choose prediction method
        if paths:
            prediction = self._causal_prediction(action, context, paths)
        else:
            logger.debug("No causal paths found, falling back to correlations")
            prediction = self._correlation_prediction(action, context)

        # APPLY: Enhance prediction
        prediction = self._apply_dynamics(prediction, context)
        prediction = self._check_invariants(prediction)
        prediction = self._calibrate_confidence(prediction, context)

        # Validate final prediction with safety validator
        if self.world_model.safety_validator:
            try:
                if hasattr(
                    self.world_model.safety_validator,
                    "validate_prediction_comprehensive",
                ):
                    pred_validation = self.world_model.safety_validator.validate_prediction_comprehensive(
                        prediction.expected,
                        prediction.lower_bound,
                        prediction.upper_bound,
                        {
                            "domain": context.domain,
                            "targets": context.targets,
                            "target_variable": (
                                context.targets[0] if context.targets else "unknown"
                            ),
                        },
                    )

                    if not pred_validation.get("safe", True):
                        logger.warning(
                            "Unsafe prediction detected, applying safety corrections: %s",
                            pred_validation.get("reason", "unknown"),
                        )
                        # Use safe values from validation
                        prediction.expected = pred_validation.get(
                            "safe_expected", prediction.expected
                        )
                        prediction.lower_bound = pred_validation.get(
                            "safe_lower", prediction.lower_bound
                        )
                        prediction.upper_bound = pred_validation.get(
                            "safe_upper", prediction.upper_bound
                        )
                        prediction.confidence *= (
                            0.5  # Reduce confidence for corrected predictions
                        )
            except Exception as e:
                logger.error(
                    "Safety validator error in validate_prediction_comprehensive: %s", e
                )

        # REMEMBER: Track prediction
        self.prediction_history.append(
            {
                "timestamp": time.time(),
                "action": action,
                "prediction": prediction,
                "context": context,
            }
        )

        return prediction

    def _causal_prediction(
        self, action: Any, context: ModelContext, paths: List[Any]
    ) -> Any:
        """
        Make prediction using causal paths.
        FIXED: Converts CausalPath objects to prediction_engine.Path objects.
        """
        from . import world_model_core

        Path = world_model_core.Path
        Prediction = world_model_core.Prediction

        if not world_model_core.PREDICTION_ENGINE_AVAILABLE or Path is None:
            logger.error(
                "Cannot make causal prediction - Path or PredictionEngine unavailable"
            )
            # Return a default Prediction
            return Prediction(
                expected=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=0.0,
                method="unavailable",
            )

        converted_paths = []
        for cp in paths:  # cp is a CausalPath
            try:
                # Get nodes and strengths from the CausalPath
                nodes = cp.nodes

                # Use .get_strengths() method which exists on CausalPath
                strengths = cp.get_strengths()

                # Create a list of edge tuples (from, to, strength)
                edges = []
                total_strength = 1.0
                for i in range(len(nodes) - 1):
                    strength = strengths[i] if i < len(strengths) else 0.0
                    edges.append((nodes[i], nodes[i + 1], strength))
                    total_strength *= strength

                # --- FIX: Construct the correct 'Path' object ---
                # prediction_engine.Path expects: nodes, edges, total_strength, confidence, evidence_types
                converted_paths.append(
                    Path(
                        nodes=nodes,
                        edges=edges,
                        total_strength=total_strength,
                        confidence=cp.confidence,
                        evidence_types=cp.evidence_types,
                    )
                )

            except Exception as e:
                logger.warning(
                    f"Failed to convert CausalPath to Path: {e}. Skipping path."
                )
                continue

        if not converted_paths and paths:
            logger.warning("All CausalPaths failed conversion to Path objects.")

        return self.world_model.ensemble_predictor.predict_with_path_ensemble(
            action,
            (
                context.to_dict() if hasattr(context, "to_dict") else context
            ),  # Pass context
            converted_paths,  # Use the new list of converted paths
        )

    def _correlation_prediction(
        self, action: Any, context: ModelContext
    ) -> Any:
        """Fallback prediction using correlations"""
        from . import world_model_core

        Prediction = world_model_core.Prediction

        if world_model_core.CorrelationTracker is None or not Prediction:
            return Prediction(
                expected=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=0.0,
                method="unavailable",
            )

        correlations = []
        for target in context.targets:
            corr_strength = self.world_model.correlation_tracker.get_correlation(
                action, target
            )
            if corr_strength:
                correlations.append((target, corr_strength))

        if not correlations:
            return Prediction(
                expected=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=0.0,
                method="no_information",
                supporting_paths=[],
            )

        predictions = []
        for target, strength in correlations:
            pred_value = strength * 1.0  # Simplified
            predictions.append(pred_value)

        expected = np.mean(predictions) if predictions else 0

        return Prediction(
            expected=expected,
            lower_bound=expected * 0.8,
            upper_bound=expected * 1.2,
            confidence=0.3,
            method="correlation_based",
            supporting_paths=[],
        )

    def _apply_dynamics(
        self, prediction: Any, context: ModelContext
    ) -> Any:
        """Apply dynamics model if temporal"""
        from . import world_model_core

        if world_model_core.DynamicsModel is None or self.world_model.dynamics is None:
            return prediction

        if context.constraints.get("time_horizon"):
            try:
                # Pass context as dict
                context_dict = (
                    context.to_dict()
                    if hasattr(context, "to_dict")
                    else context.__dict__.copy()
                )  # Use __dict__ fallback
                return self.world_model.dynamics.apply(
                    prediction, context_dict, context.constraints["time_horizon"]
                )
            except Exception as e:
                logger.warning(f"Failed to apply dynamics to prediction: {e}")
                return prediction
        return prediction

    def _check_invariants(self, prediction: Any) -> Any:
        """Check invariant violations"""
        from . import world_model_core

        if not world_model_core.INVARIANT_DETECTOR_AVAILABLE or self.world_model.invariants is None:
            return prediction

        # Convert prediction.expected to dict format for invariant checker
        state = (
            {"value": prediction.expected}
            if isinstance(prediction.expected, (int, float))
            else prediction.expected
        )

        violations = self.world_model.invariants.check_invariant_violations(state)
        if violations:
            logger.warning("Prediction violates invariants: %s", violations)
            prediction.confidence *= 0.5  # Reduce confidence
        return prediction

    def _calibrate_confidence(
        self, prediction: Any, context: ModelContext
    ) -> Any:
        """Calibrate prediction confidence"""
        from . import world_model_core

        if (
            not world_model_core.CONFIDENCE_CALIBRATOR_AVAILABLE
            or self.world_model.confidence_calibrator is None
        ):
            return prediction

        calibrated = self.world_model.confidence_calibrator.calibrate(
            prediction.confidence, context.features
        )
        prediction.confidence = calibrated
        return prediction
