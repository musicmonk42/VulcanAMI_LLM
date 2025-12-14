# domain_validators.py
"""
Domain-specific safety validators for VULCAN-AGI Safety Module.
Implements specialized validators for different operational domains.

Revision / Fix Notes (Applied):
1. Removed merge conflict markers (<<<<<<< HEAD / ======= / >>>>>>>).
2. Unified initialization logic with thread-safe lock (_DOMAIN_VALIDATORS_LOCK).
3. Preserved all original classes and logic unmodified except conflict resolution.
4. Ensured validator_registry global is created exactly once with locking.
5. No truncation; full file content retained.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DOMAIN_VALIDATORS_INIT_DONE = False
_DOMAIN_VALIDATORS_LOCK = threading.RLock()

# ============================================================
# DOMAIN VALIDATOR BASE
# ============================================================


class ValidationResult:
    """Result of a domain validation check."""

    def __init__(
        self,
        safe: bool,
        reason: str = "",
        corrected_values: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.safe = safe
        self.reason = reason
        self.corrected_values = corrected_values or {}
        self.severity = severity
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "safe": self.safe,
            "reason": self.reason,
            "corrected_values": self.corrected_values,
            "severity": self.severity,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class DomainValidator:
    """Base class for domain-specific validators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_history = deque(maxlen=1000)
        self.violation_counts = defaultdict(int)

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate data for domain-specific safety."""
        raise NotImplementedError("Subclasses must implement validate()")

    def _record_validation(self, result: ValidationResult, data: Any):
        """Record validation result."""
        self.validation_history.append(
            {
                "result": result,
                "data_summary": str(data)[:100],
                "timestamp": time.time(),
            }
        )

        if not result.safe:
            self.violation_counts[result.reason] += 1

    def _max_severity(self, current: str, new: str) -> str:
        """Return the more severe of two severity levels."""
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        current_level = severity_levels.get(current, 0)
        new_level = severity_levels.get(new, 0)
        return current if current_level >= new_level else new

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}

        total = len(self.validation_history)
        safe = sum(1 for v in self.validation_history if v["result"].safe)

        return {
            "total_validations": total,
            "safe_count": safe,
            "unsafe_count": total - safe,
            "safety_rate": safe / total if total > 0 else 0,
            "top_violations": dict(
                sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ),
        }


# ============================================================
# CAUSAL SAFETY VALIDATOR
# ============================================================


class CausalSafetyValidator(DomainValidator):
    """Validator for causal reasoning operations."""

    def __init__(
        self,
        safe_regions: Optional[Dict[str, Tuple[float, float]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.safe_regions = safe_regions or {}

        # Unsafe causal patterns that indicate harm amplification
        self.unsafe_patterns = {
            "harm->increase",
            "danger->amplify",
            "risk->escalate",
            "threat->magnify",
            "damage->compound",
            "injury->worsen",
            "violence->spread",
            "toxicity->multiply",
            "failure->cascade",
            "error->propagate",
        }

        # Maximum safe values
        self.max_causal_strength = self.config.get("max_causal_strength", 10.0)
        self.max_path_length = self.config.get("max_path_length", 20)
        self.max_total_amplification = self.config.get("max_total_amplification", 10.0)

        logger.info("CausalSafetyValidator initialized")

    def validate_causal_edge(
        self,
        cause: str,
        effect: str,
        strength: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a causal edge.

        Args:
            cause: Cause variable name
            effect: Effect variable name
            strength: Causal strength (coefficient)
            context: Additional context

        Returns:
            ValidationResult
        """
        violations = []
        corrected = {}
        severity = "medium"

        # Check for NaN or Inf (CRITICAL - check first)
        if np.isnan(strength) or np.isinf(strength):
            violations.append("Causal strength is NaN or Inf")
            corrected["strength"] = 0.0
            severity = "critical"

        # Check for empty names (CRITICAL)
        if not cause or not effect:
            violations.append("Cause or effect name is empty")
            severity = self._max_severity(severity, "critical")

        # Check for excessive causal strength
        if abs(strength) > self.max_causal_strength:
            violations.append(f"Causal strength too large: {strength:.2f}")
            corrected["strength"] = np.clip(
                strength, -self.max_causal_strength, self.max_causal_strength
            )
            severity = self._max_severity(severity, "high")

        # Check for unsafe patterns
        edge_pattern = f"{cause.lower()}->{effect.lower()}"
        for unsafe_pattern in self.unsafe_patterns:
            if unsafe_pattern in edge_pattern:
                violations.append(f"Edge matches unsafe pattern: {unsafe_pattern}")
                severity = self._max_severity(severity, "high")

        # Check for self-loops
        if cause == effect:
            violations.append("Self-loop detected (variable causing itself)")
            severity = self._max_severity(severity, "high")

        # Check for very small non-zero strengths (numerical instability)
        if 0 < abs(strength) < 1e-10:
            violations.append(
                f"Causal strength too small (numerical instability): {strength}"
            )
            corrected["strength"] = 0.0
            severity = self._max_severity(severity, "low")

        # Check for safe regions
        if context:
            if "variable_bounds" in context:
                bounds = context["variable_bounds"]
                if cause in bounds and effect in bounds:
                    cause_range = bounds[cause][1] - bounds[cause][0]
                    effect_range = bounds[effect][1] - bounds[effect][0]

                    # Check if causal effect would push outside safe region
                    implied_change = abs(strength * cause_range)
                    if implied_change > effect_range:
                        violations.append(
                            f"Causal effect could push {effect} outside safe region"
                        )
                        severity = self._max_severity(severity, "high")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata={"cause": cause, "effect": effect, "strength": strength},
            )
        else:
            result = ValidationResult(
                safe=True,
                metadata={"cause": cause, "effect": effect, "strength": strength},
            )

        self._record_validation(
            result, {"cause": cause, "effect": effect, "strength": strength}
        )
        return result

    def validate_causal_path(
        self,
        nodes: List[str],
        strengths: List[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a causal path.

        Args:
            nodes: List of node names in path
            strengths: List of causal strengths along path
            context: Additional context

        Returns:
            ValidationResult
        """
        violations = []
        corrected = {}
        severity = "medium"
        total_strength = 1.0

        # Check if nodes and strengths match (CRITICAL)
        if len(strengths) != len(nodes) - 1:
            violations.append(
                f"Mismatch: {len(nodes)} nodes but {len(strengths)} strengths "
                f"(expected {len(nodes) - 1})"
            )
            severity = "critical"

        # Check for empty paths (CRITICAL)
        if not nodes:
            violations.append("Path is empty")
            severity = self._max_severity(severity, "critical")

        # Calculate total amplification and check for NaN/Inf (CRITICAL)
        if strengths:
            for strength in strengths:
                if np.isnan(strength) or np.isinf(strength):
                    violations.append("Path contains NaN or Inf strength")
                    severity = self._max_severity(severity, "critical")
                    break
                total_strength *= strength

            if abs(total_strength) > self.max_total_amplification:
                violations.append(
                    f"Path shows excessive amplification: {total_strength:.2f}x "
                    f"(max: {self.max_total_amplification}x)"
                )
                corrected["safe_amplification"] = np.clip(
                    total_strength,
                    -self.max_total_amplification,
                    self.max_total_amplification,
                )
                severity = self._max_severity(severity, "high")

        # Check path length
        if len(nodes) > self.max_path_length:
            violations.append(
                f"Path too long: {len(nodes)} nodes (max: {self.max_path_length})"
            )
            severity = self._max_severity(severity, "high")

        # Check for cycles
        if len(nodes) != len(set(nodes)):
            violations.append("Path contains cycles")
            severity = self._max_severity(severity, "high")

        # Check each edge in the path
        edge_violations = []
        for i in range(len(nodes) - 1):
            edge_result = self.validate_causal_edge(
                nodes[i],
                nodes[i + 1],
                strengths[i] if i < len(strengths) else 0.0,
                context,
            )
            if not edge_result.safe:
                edge_violations.append(
                    f"Edge {nodes[i]}->{nodes[i + 1]}: {edge_result.reason}"
                )
                # Propagate the most severe edge severity
                severity = self._max_severity(severity, edge_result.severity)

        if edge_violations:
            violations.extend(edge_violations)

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata={
                    "path_length": len(nodes),
                    "total_amplification": total_strength if strengths else 0,
                },
            )
        else:
            result = ValidationResult(
                safe=True,
                metadata={
                    "path_length": len(nodes),
                    "total_amplification": total_strength if strengths else 0,
                },
            )

        self._record_validation(result, {"nodes": nodes, "strengths": strengths})
        return result

    def validate_causal_graph(
        self,
        adjacency: Dict[str, List[Tuple[str, float]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate entire causal graph.

        Args:
            adjacency: Adjacency list {node: [(child, strength), ...]}
            context: Additional context

        Returns:
            ValidationResult
        """
        violations = []
        severity = "medium"

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            if node in adjacency:
                for child, _ in adjacency[node]:
                    if child not in visited:
                        if has_cycle(child, visited, rec_stack):
                            return True
                    elif child in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        visited = set()
        for node in adjacency:
            if node not in visited:
                if has_cycle(node, visited, set()):
                    violations.append("Graph contains cycles")
                    severity = self._max_severity(severity, "high")
                    break

        # Validate each edge
        edge_violations = 0
        for parent, children in adjacency.items():
            for child, strength in children:
                edge_result = self.validate_causal_edge(
                    parent, child, strength, context
                )
                if not edge_result.safe:
                    edge_violations += 1
                    # Propagate the most severe edge severity
                    severity = self._max_severity(severity, edge_result.severity)

        if edge_violations > 0:
            violations.append(f"Graph has {edge_violations} unsafe edges")
            # Severity already set by edges

        # Check graph size
        total_edges = sum(len(children) for children in adjacency.values())
        max_edges = self.config.get("max_graph_edges", 1000)
        if total_edges > max_edges:
            violations.append(
                f"Graph too large: {total_edges} edges (max: {max_edges})"
            )
            severity = self._max_severity(severity, "high")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                severity=severity,
                metadata={
                    "node_count": len(adjacency),
                    "edge_count": total_edges,
                    "edge_violations": edge_violations,
                },
            )
        else:
            result = ValidationResult(
                safe=True,
                metadata={"node_count": len(adjacency), "edge_count": total_edges},
            )

        self._record_validation(result, {"adjacency": adjacency})
        return result

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Overridden validate method for CausalSafetyValidator."""
        if (
            isinstance(data, dict)
            and "cause" in data
            and "effect" in data
            and "strength" in data
        ):
            return self.validate_causal_edge(
                data["cause"], data["effect"], data["strength"], context
            )
        elif isinstance(data, dict) and "nodes" in data and "strengths" in data:
            return self.validate_causal_path(data["nodes"], data["strengths"], context)
        elif isinstance(data, dict):
            # Assuming it's an adjacency list
            return self.validate_causal_graph(data, context)

        return ValidationResult(
            safe=False, reason="Unknown causal data format", severity="critical"
        )


# ============================================================
# PREDICTION SAFETY VALIDATOR
# ============================================================


class PredictionSafetyValidator(DomainValidator):
    """Validator for prediction operations."""

    def __init__(
        self,
        safe_regions: Optional[Dict[str, Tuple[float, float]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.safe_regions = safe_regions or {}

        # Configuration
        self.max_magnitude = self.config.get("max_magnitude", 1000.0)
        self.max_uncertainty_ratio = self.config.get("max_uncertainty_ratio", 5.0)
        self.min_confidence = self.config.get("min_confidence", 0.1)

        logger.info("PredictionSafetyValidator initialized")

    def validate_prediction(
        self,
        expected: float,
        lower: float,
        upper: float,
        variable: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate prediction value and bounds.

        Args:
            expected: Expected/mean prediction
            lower: Lower confidence bound
            upper: Upper confidence bound
            variable: Variable name being predicted
            context: Additional context

        Returns:
            ValidationResult
        """
        violations = []
        corrected = {}
        severity = "medium"

        safe_expected = expected
        safe_lower = lower
        safe_upper = upper

        # Check for NaN or Inf (CRITICAL - check first)
        if np.isnan(expected) or np.isinf(expected):
            violations.append("Prediction contains NaN or Inf")
            safe_expected = 0.0
            safe_lower = 0.0
            safe_upper = 0.0
            severity = "critical"
        elif np.isnan(lower) or np.isinf(lower) or np.isnan(upper) or np.isinf(upper):
            violations.append("Prediction bounds contain NaN or Inf")
            safe_lower = expected - abs(expected) * 0.1
            safe_upper = expected + abs(expected) * 0.1
            severity = self._max_severity(severity, "high")

        # Check magnitude
        if abs(expected) > self.max_magnitude:
            violations.append(f"Prediction magnitude too large: {expected:.2f}")
            safe_expected = np.clip(expected, -self.max_magnitude, self.max_magnitude)
            safe_lower = np.clip(lower, -self.max_magnitude, self.max_magnitude)
            safe_upper = np.clip(upper, -self.max_magnitude, self.max_magnitude)
            severity = self._max_severity(severity, "high")

        # Check uncertainty
        uncertainty = upper - lower
        if uncertainty < 0:
            violations.append(
                f"Invalid bounds: lower ({lower:.2f}) > upper ({upper:.2f})"
            )
            safe_lower = min(lower, upper)
            safe_upper = max(lower, upper)
            severity = self._max_severity(severity, "high")
        elif uncertainty > abs(expected) * self.max_uncertainty_ratio:
            violations.append(
                f"Prediction too uncertain: range {uncertainty:.2f} "
                f"(> {self.max_uncertainty_ratio}x expected value)"
            )
            safe_lower = expected - abs(expected)
            safe_upper = expected + abs(expected)
            severity = self._max_severity(severity, "medium")

        # Check if expected is within bounds
        if not (lower <= expected <= upper):
            violations.append(
                f"Expected value {expected:.2f} not within bounds [{lower:.2f}, {upper:.2f}]"
            )
            safe_expected = np.clip(expected, lower, upper)
            severity = self._max_severity(severity, "high")

        # Check safe regions
        if variable in self.safe_regions:
            min_val, max_val = self.safe_regions[variable]

            if not (min_val <= expected <= max_val):
                violations.append(
                    f"Prediction outside safe region for {variable}: "
                    f"{expected:.2f} not in [{min_val:.2f}, {max_val:.2f}]"
                )
                safe_expected = np.clip(expected, min_val, max_val)
                safe_lower = np.clip(lower, min_val, max_val)
                safe_upper = np.clip(upper, min_val, max_val)
                severity = self._max_severity(severity, "high")

            # Check if bounds extend outside safe region
            if lower < min_val or upper > max_val:
                violations.append(
                    f"Prediction bounds extend outside safe region for {variable}"
                )
                safe_lower = max(safe_lower, min_val)
                safe_upper = min(safe_upper, max_val)
                severity = self._max_severity(severity, "medium")

        # Add corrected values
        if violations:
            corrected["safe_expected"] = safe_expected
            corrected["safe_lower"] = safe_lower
            corrected["safe_upper"] = safe_upper

        # Check context-specific constraints
        if context:
            if "critical_variable" in context and context["critical_variable"]:
                if uncertainty > abs(expected) * 0.5:
                    violations.append(
                        f"Critical variable {variable} has excessive uncertainty"
                    )
                    severity = self._max_severity(severity, "high")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata={
                    "variable": variable,
                    "expected": expected,
                    "uncertainty": uncertainty,
                },
            )
        else:
            result = ValidationResult(
                safe=True,
                metadata={
                    "variable": variable,
                    "expected": expected,
                    "uncertainty": uncertainty,
                },
            )

        self._record_validation(
            result,
            {
                "expected": expected,
                "lower": lower,
                "upper": upper,
                "variable": variable,
            },
        )
        return result

    def validate_prediction_batch(
        self,
        predictions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate batch of predictions.

        Args:
            predictions: List of predictions with 'expected', 'lower', 'upper', 'variable'
            context: Additional context

        Returns:
            ValidationResult
        """
        violations = []
        unsafe_count = 0
        corrected_predictions = []
        severity = "medium"

        for i, pred in enumerate(predictions):
            result = self.validate_prediction(
                pred.get("expected", 0.0),
                pred.get("lower", 0.0),
                pred.get("upper", 0.0),
                pred.get("variable", f"var_{i}"),
                context,
            )

            if not result.safe:
                unsafe_count += 1
                violations.append(
                    f"Prediction {i} ({pred.get('variable', f'var_{i}')}): {result.reason}"
                )
                corrected_predictions.append(result.corrected_values)
                severity = self._max_severity(severity, result.severity)
            else:
                corrected_predictions.append(pred)

        if unsafe_count > len(predictions) * 0.3:
            severity = self._max_severity(severity, "high")

        if violations:
            result = ValidationResult(
                safe=False,
                reason=f"{unsafe_count}/{len(predictions)} unsafe predictions; "
                + "; ".join(violations[:5]),
                corrected_values={"predictions": corrected_predictions},
                severity=severity,
                metadata={"total": len(predictions), "unsafe": unsafe_count},
            )
        else:
            result = ValidationResult(
                safe=True, metadata={"total": len(predictions), "unsafe": 0}
            )

        self._record_validation(result, {"predictions": predictions})
        return result

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Overridden validate method for PredictionSafetyValidator."""
        if isinstance(data, list):
            return self.validate_prediction_batch(data, context)
        elif isinstance(data, dict) and "expected" in data and "variable" in data:
            return self.validate_prediction(
                data.get("expected", 0.0),
                data.get("lower", data.get("expected", 0.0)),
                data.get("upper", data.get("expected", 0.0)),
                data["variable"],
                context,
            )
        return ValidationResult(
            safe=False, reason="Unknown prediction data format", severity="critical"
        )


# ============================================================
# OPTIMIZATION SAFETY VALIDATOR
# ============================================================


class OptimizationSafetyValidator(DomainValidator):
    """Validator for optimization operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_iterations = self.config.get("max_iterations", 10000)
        self.min_improvement = self.config.get("min_improvement", 1e-8)
        self.max_objective_value = self.config.get("max_objective_value", 1e10)

        logger.info("OptimizationSafetyValidator initialized")

    def validate_optimization_params(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate optimization parameters."""
        violations = []
        corrected = {}
        severity = "medium"

        required = ["max_iterations", "tolerance"]
        for req in required:
            if req not in params:
                violations.append(f"Missing required parameter: {req}")
                severity = self._max_severity(severity, "high")

        max_iter = params.get("max_iterations", 100)
        if max_iter > self.max_iterations:
            violations.append(
                f"Too many iterations: {max_iter} (max: {self.max_iterations})"
            )
            corrected["max_iterations"] = self.max_iterations
            severity = self._max_severity(severity, "high")

        learning_rate = params.get("learning_rate", 0.01)
        if learning_rate <= 0 or learning_rate > 1.0:
            violations.append(f"Invalid learning rate: {learning_rate}")
            corrected["learning_rate"] = 0.01
            severity = self._max_severity(severity, "high")

        tolerance = params.get("tolerance", 1e-6)
        if tolerance < self.min_improvement:
            violations.append(
                f"Tolerance too small: {tolerance} (min: {self.min_improvement})"
            )
            corrected["tolerance"] = self.min_improvement
            severity = self._max_severity(severity, "medium")

        if "bounds" in params:
            bounds = params["bounds"]
            for var, (lb, ub) in bounds.items():
                if lb >= ub:
                    violations.append(
                        f"Invalid bounds for {var}: lower ({lb}) >= upper ({ub})"
                    )
                    corrected[f"bounds_{var}"] = (lb, lb + 1.0)
                    severity = self._max_severity(severity, "high")
                if np.isinf(lb) or np.isinf(ub):
                    violations.append(f"Infinite bounds for {var}")
                    severity = self._max_severity(severity, "medium")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata={"params": params},
            )
        else:
            result = ValidationResult(safe=True, metadata={"params": params})

        self._record_validation(result, params)
        return result

    def validate_optimization_result(
        self, result_obj: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate optimization result."""
        violations = []
        severity = "medium"

        if not result_obj.get("success", False):
            violations.append(
                f"Optimization failed: {result_obj.get('message', 'Unknown reason')}"
            )
            severity = self._max_severity(severity, "high")

        obj_value = result_obj.get("objective_value", 0)
        if np.isnan(obj_value) or np.isinf(obj_value):
            violations.append("Objective value is NaN or Inf")
            severity = self._max_severity(severity, "critical")
        elif abs(obj_value) > self.max_objective_value:
            violations.append(f"Objective value too large: {obj_value}")
            severity = self._max_severity(severity, "high")

        solution = result_obj.get("solution", {})
        for var, val in solution.items():
            if np.isnan(val) or np.isinf(val):
                violations.append(f"Solution variable {var} is NaN or Inf")
                severity = self._max_severity(severity, "critical")

        iterations = result_obj.get("iterations", 0)
        if iterations >= self.max_iterations:
            violations.append(f"Optimization hit iteration limit: {iterations}")
            severity = self._max_severity(severity, "medium")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                severity=severity,
                metadata={"result": result_obj},
            )
        else:
            result = ValidationResult(safe=True, metadata={"result": result_obj})

        self._record_validation(result, result_obj)
        return result

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Overridden validate method for OptimizationSafetyValidator."""
        if isinstance(data, dict):
            if "max_iterations" in data or "tolerance" in data:
                return self.validate_optimization_params(data, context)
            elif "success" in data or "objective_value" in data:
                return self.validate_optimization_result(data, context)

        return ValidationResult(
            safe=False, reason="Unknown optimization data format", severity="critical"
        )


# ============================================================
# DATA PROCESSING SAFETY VALIDATOR
# ============================================================


class DataProcessingSafetyValidator(DomainValidator):
    """Validator for data processing operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_data_size_mb = self.config.get("max_data_size_mb", 1000)
        self.max_rows = self.config.get("max_rows", 10000000)
        self.max_columns = self.config.get("max_columns", 10000)
        self.max_missing_ratio = self.config.get("max_missing_ratio", 0.5)

        logger.info("DataProcessingSafetyValidator initialized")

    def validate_dataframe(
        self, df_info: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate dataframe safety."""
        violations = []
        corrected = {}
        severity = "medium"

        rows = df_info.get("rows", 0)
        cols = df_info.get("columns", 0)

        if rows > self.max_rows:
            violations.append(f"Too many rows: {rows} (max: {self.max_rows})")
            corrected["max_rows"] = self.max_rows
            severity = self._max_severity(severity, "high")

        if cols > self.max_columns:
            violations.append(f"Too many columns: {cols} (max: {self.max_columns})")
            corrected["max_columns"] = self.max_columns
            severity = self._max_severity(severity, "high")

        memory_mb = df_info.get("memory_mb", 0)
        if memory_mb > self.max_data_size_mb:
            violations.append(
                f"Data too large: {memory_mb}MB (max: {self.max_data_size_mb}MB)"
            )
            severity = self._max_severity(severity, "high")

        missing_ratio = df_info.get("missing_ratio", 0)
        if missing_ratio > self.max_missing_ratio:
            violations.append(f"Too much missing data: {missing_ratio * 100:.1f}%")
            severity = self._max_severity(severity, "medium")

        dtypes = df_info.get("dtypes", {})
        for col, dtype in dtypes.items():
            if dtype == "object" and rows > 100000:
                violations.append(f"Large object column {col} may cause memory issues")
                severity = self._max_severity(severity, "medium")

        if "duplicate_columns" in df_info and df_info["duplicate_columns"]:
            violations.append(f"Duplicate columns: {df_info['duplicate_columns']}")
            severity = self._max_severity(severity, "low")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata=df_info,
            )
        else:
            result = ValidationResult(safe=True, metadata=df_info)

        self._record_validation(result, df_info)
        return result

    def validate_data_transformation(
        self, transform_info: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate data transformation operation."""
        violations = []
        severity = "medium"

        transform_type = transform_info.get("type", "unknown")
        unsafe_transforms = ["eval", "exec", "compile"]

        if transform_type in unsafe_transforms:
            violations.append(f"Unsafe transformation type: {transform_type}")
            severity = "critical"

        input_rows = transform_info.get("input_rows", 0)
        output_rows = transform_info.get("output_rows", 0)

        if output_rows == 0 and input_rows > 0:
            violations.append("Transformation eliminated all data")
            severity = self._max_severity(severity, "high")

        if output_rows > input_rows * 10:
            violations.append(
                f"Transformation dramatically expanded data: {input_rows} -> {output_rows}"
            )
            severity = self._max_severity(severity, "medium")

        if "columns_dropped" in transform_info:
            dropped = len(transform_info["columns_dropped"])
            total = transform_info.get("total_columns", 0)
            if dropped > total * 0.5:
                violations.append(f"Transformation dropped {dropped}/{total} columns")
                severity = self._max_severity(severity, "medium")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                severity=severity,
                metadata=transform_info,
            )
        else:
            result = ValidationResult(safe=True, metadata=transform_info)

        self._record_validation(result, transform_info)
        return result

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Overridden validate method for DataProcessingSafetyValidator."""
        if isinstance(data, dict):
            if "rows" in data and "columns" in data:
                return self.validate_dataframe(data, context)
            elif "type" in data and "input_rows" in data:
                return self.validate_data_transformation(data, context)

        return ValidationResult(
            safe=False, reason="Unknown data processing format", severity="critical"
        )


# ============================================================
# MODEL INFERENCE VALIDATOR (SIMPLIFIED)
# ============================================================


class ModelInferenceValidator:
    """
    Safety validator for model inference operations
    Validates predictions, confidence bounds, and output consistency
    """

    def __init__(self, safety_config: Optional[Dict[str, Any]] = None):
        self.config = safety_config or {}
        self.min_confidence = self.config.get("min_confidence", 0.1)
        self.max_output_range = self.config.get("max_output_range", 1e6)
        logger.info("ModelInferenceValidator initialized")

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        violations = []

        if "prediction" not in data and "output" not in data:
            return {
                "safe": True,
                "violations": [],
                "reason": "No prediction to validate",
            }

        prediction = data.get("prediction") or data.get("output")

        if "confidence" in data:
            confidence = data["confidence"]
            if not isinstance(confidence, (int, float)):
                violations.append(
                    {
                        "type": "invalid_confidence_type",
                        "severity": "high",
                        "message": f"Confidence must be numeric, got {type(confidence)}",
                    }
                )
            elif confidence < 0 or confidence > 1:
                violations.append(
                    {
                        "type": "confidence_out_of_range",
                        "severity": "high",
                        "message": f"Confidence {confidence} outside [0, 1] range",
                    }
                )
            elif confidence < self.min_confidence:
                violations.append(
                    {
                        "type": "confidence_too_low",
                        "severity": "medium",
                        "message": f"Confidence {confidence} below threshold {self.min_confidence}",
                    }
                )

        if isinstance(prediction, (int, float)):
            if abs(prediction) > self.max_output_range:
                violations.append(
                    {
                        "type": "prediction_out_of_bounds",
                        "severity": "high",
                        "message": f"Prediction {prediction} exceeds range ±{self.max_output_range}",
                    }
                )
            if prediction != prediction:
                violations.append(
                    {
                        "type": "prediction_nan",
                        "severity": "critical",
                        "message": "Prediction is NaN",
                    }
                )
            elif abs(prediction) == float("inf"):
                violations.append(
                    {
                        "type": "prediction_infinite",
                        "severity": "critical",
                        "message": "Prediction is infinite",
                    }
                )

        if "lower_bound" in data and "upper_bound" in data:
            lower = data["lower_bound"]
            upper = data["upper_bound"]
            if lower > upper:
                violations.append(
                    {
                        "type": "invalid_bounds",
                        "severity": "critical",
                        "message": f"Lower bound {lower} > upper bound {upper}",
                    }
                )
            if isinstance(prediction, (int, float)):
                if prediction < lower or prediction > upper:
                    violations.append(
                        {
                            "type": "prediction_outside_bounds",
                            "severity": "high",
                            "message": f"Prediction {prediction} outside bounds [{lower}, {upper}]",
                        }
                    )

        if "expected" in data and isinstance(prediction, (int, float)):
            expected = data["expected"]
            if isinstance(expected, (int, float)):
                deviation = abs(prediction - expected) / (abs(expected) + 1e-10)
                if deviation > 10.0:
                    violations.append(
                        {
                            "type": "large_deviation",
                            "severity": "medium",
                            "message": f"Large deviation from expected: {deviation:.2f}x",
                        }
                    )

        critical_violations = [v for v in violations if v["severity"] == "critical"]
        safe = len(critical_violations) == 0

        return {
            "safe": safe,
            "violations": violations,
            "reason": (
                critical_violations[0]["message"]
                if critical_violations
                else "Validation passed"
            ),
        }


# ============================================================
# MODEL INFERENCE SAFETY VALIDATOR (COMPREHENSIVE)
# ============================================================


class ModelInferenceSafetyValidator(DomainValidator):
    """Validator for model inference operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_batch_size = self.config.get("max_batch_size", 10000)
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.max_inference_time = self.config.get("max_inference_time", 60.0)

        logger.info("ModelInferenceSafetyValidator initialized")

    def validate_inference_input(
        self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        violations = []
        corrected = {}
        severity = "medium"

        batch_size = input_data.get("batch_size", 1)
        if batch_size > self.max_batch_size:
            violations.append(
                f"Batch size too large: {batch_size} (max: {self.max_batch_size})"
            )
            corrected["batch_size"] = self.max_batch_size
            severity = self._max_severity(severity, "high")

        shape = input_data.get("shape", ())
        if len(shape) == 0:
            violations.append("Input has no shape")
            severity = self._max_severity(severity, "high")

        if input_data.get("contains_nan", False):
            violations.append("Input contains NaN values")
            severity = self._max_severity(severity, "high")

        if input_data.get("contains_inf", False):
            violations.append("Input contains Inf values")
            severity = self._max_severity(severity, "high")

        if "min_value" in input_data and "max_value" in input_data:
            min_val = input_data["min_value"]
            max_val = input_data["max_value"]
            if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                violations.append(
                    f"Input values have extreme range: [{min_val}, {max_val}]"
                )
                severity = self._max_severity(severity, "medium")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                corrected_values=corrected,
                severity=severity,
                metadata=input_data,
            )
        else:
            result = ValidationResult(safe=True, metadata=input_data)

        self._record_validation(result, input_data)
        return result

    def validate_inference_output(
        self, output_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        violations = []
        severity = "medium"

        if output_data.get("contains_nan", False):
            violations.append("Output contains NaN values")
            severity = "critical"

        if output_data.get("contains_inf", False):
            violations.append("Output contains Inf values")
            severity = self._max_severity(severity, "critical")

        if "shape" in output_data:
            shape = output_data["shape"]
            if len(shape) == 0 or shape[0] == 0:
                violations.append("Output has invalid shape")
                severity = self._max_severity(severity, "high")

        if "confidences" in output_data:
            confidences = output_data["confidences"]
            if isinstance(confidences, (list, np.ndarray)):
                min_conf = np.min(confidences)
                if min_conf < self.min_confidence:
                    violations.append(f"Low confidence predictions: min={min_conf:.3f}")
                    severity = self._max_severity(severity, "medium")

        inference_time = output_data.get("inference_time", 0)
        if inference_time > self.max_inference_time:
            violations.append(f"Inference took too long: {inference_time:.2f}s")
            severity = self._max_severity(severity, "medium")

        if violations:
            result = ValidationResult(
                safe=False,
                reason="; ".join(violations),
                severity=severity,
                metadata=output_data,
            )
        else:
            result = ValidationResult(safe=True, metadata=output_data)

        self._record_validation(result, output_data)
        return result

    def validate(
        self, data: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        if isinstance(data, dict):
            if "batch_size" in data or "shape" in data:
                return self.validate_inference_input(data, context)
            elif "confidences" in data or "inference_time" in data:
                return self.validate_inference_output(data, context)

        return ValidationResult(
            safe=False,
            reason="Unknown model inference data format",
            severity="critical",
        )


# ============================================================
# DOMAIN VALIDATOR REGISTRY
# ============================================================


class DomainValidatorRegistry:
    """Registry for managing domain validators."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._validators = {}
        logger.info("DomainValidatorRegistry initialized")

    def register(self, name: str, validator_obj):
        if name not in self._validators:
            self._validators[name] = validator_obj
            logger.info(f"Registered validator for domain: {name}")
        else:
            logger.debug(f"Skipped duplicate domain validator: {name}")

    def list_domains(self):
        return sorted(self._validators.keys())

    def get_validator(self, domain: str, **kwargs) -> Optional[DomainValidator]:
        if domain in self._validators:
            return self._validators[domain]
        logger.warning(f"No validator registered for domain: {domain}")
        return None

    def validate(
        self, domain: str, data: Any, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ValidationResult:
        validator = self.get_validator(domain, **kwargs)
        if validator:
            return validator.validate(data, context)
        return ValidationResult(
            safe=False,
            reason=f"No validator available for domain: {domain}",
            severity="high",
        )


def initialize_domain_validators():
    global _DOMAIN_VALIDATORS_INIT_DONE
    with _DOMAIN_VALIDATORS_LOCK:
        if _DOMAIN_VALIDATORS_INIT_DONE:
            logger.debug("Domain validators already initialized – skipping.")
            return DomainValidatorRegistry()

        registry = DomainValidatorRegistry()
        registry.register("causal", CausalSafetyValidator())
        registry.register("prediction", PredictionSafetyValidator())
        registry.register("optimization", OptimizationSafetyValidator())
        registry.register("data_processing", DataProcessingSafetyValidator())
        registry.register("model_inference", ModelInferenceSafetyValidator())

        logger.info("Domain validators initialized")
        _DOMAIN_VALIDATORS_INIT_DONE = True
        return registry


# Global registry instance
validator_registry = initialize_domain_validators()
