"""
validation_engine.py - Validation engine for Knowledge Crystallizer
Part of the VULCAN-AGI system

IMPLEMENTATION COMPLETE:
1. Added Principle dataclass with execution_logic field
2. Implemented real execution in _create_test_script()
3. Removed not_implemented flag
4. Added execution result comparison and scoring
5. Enhanced sandboxing and security
6. Added thread safety and validation improvements
"""

import copy
import hashlib
import inspect
import json
import logging
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# Platform-specific imports with fallbacks
try:
    import resource

    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    logging.warning(
        "resource module not available (Windows?), resource limits will be disabled"
    )

try:
    import signal

    SIGNAL_AVAILABLE = True
except ImportError:
    SIGNAL_AVAILABLE = False
    logging.warning("signal module not available, timeout handling will be limited")

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation"""

    BASIC = "basic"
    CONSISTENCY = "consistency"
    DOMAIN_SPECIFIC = "domain_specific"
    GENERALIZATION = "generalization"
    COMPREHENSIVE = "comprehensive"


class DomainCategory(Enum):
    """Categories of domains by data availability"""

    HIGH_DATA = "high_data"
    MEDIUM_DATA = "medium_data"
    LOW_DATA = "low_data"
    NO_DATA = "no_data"


class TestResult(Enum):
    """Test execution results"""

    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class Principle:
    """
    Principle with executable logic

    This is the core dataclass for principles that can be validated.
    execution_logic should be a callable that takes inputs and returns outputs.
    """

    id: str
    core_pattern: Any
    confidence: float
    applicable_domains: List[str] = field(default_factory=list)
    contraindicated_domains: List[str] = field(default_factory=list)
    measurement_requirements: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    baseline_performance: float = 0.5
    origin_domain: Optional[str] = None

    # Execution logic - can be one of:
    # 1. A callable function: Callable[[Dict[str, Any]], Dict[str, Any]]
    # 2. A code string: str (Python code to execute)
    # 3. A rule specification: Dict[str, Any] (rule-based execution)
    execution_logic: Optional[Any] = None
    execution_type: str = "function"  # "function", "code_string", or "rule_spec"

    # Optional: pre-computed expected outputs for validation
    compute_expected: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate principle after initialization"""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Principle id must be non-empty string")

        if not isinstance(self.confidence, (int, float)) or not (
            0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence must be a number between 0 and 1")

        if self.execution_type not in ["function", "code_string", "rule_spec"]:
            raise ValueError(f"Invalid execution_type: {self.execution_type}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the principle with given inputs

        Args:
            inputs: Input data dictionary

        Returns:
            Output data dictionary
        """
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary")

        if self.execution_logic is None:
            raise NotImplementedError(
                f"Principle {self.id} has no execution logic defined"
            )

        if self.execution_type == "function":
            if callable(self.execution_logic):
                return self.execution_logic(inputs)
            else:
                raise ValueError(
                    f"Principle {self.id} execution_type is 'function' but execution_logic is not callable"
                )

        elif self.execution_type == "code_string":
            # Execute code string in RESTRICTED namespace
            safe_builtins = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "type": type,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sorted": sorted,
                    "reversed": reversed,
                    "map": map,
                    "filter": filter,
                    "any": any,
                    "all": all,
                    "round": round,
                    "pow": pow,
                    "True": True,
                    "False": False,
                    "None": None,
                }
            }

            # Validate code doesn't have dangerous operations
            # Use word boundaries to avoid false positives (e.g., 'input' as a dict key)
            dangerous_keywords = [
                "import",
                "__import__",
                "eval",
                "exec",
                "compile",
                "open",
                "file",
                "__builtins__",
                "globals",
                "locals",
                "vars",
                "dir",
                "getattr",
                "setattr",
                "delattr",
                "input",
                "raw_input",
            ]

            code_str = str(self.execution_logic)
            for keyword in dangerous_keywords:
                # Check for keyword as a standalone word or function call (not in strings)
                # Pattern matches: keyword( or keyword. or keyword[ or keyword as a word boundary
                pattern = r"\b" + re.escape(keyword) + r"(?:\s*[(\.\[]|\s|$)"
                if re.search(pattern, code_str, re.IGNORECASE):
                    raise ValueError(
                        f"Dangerous operation '{keyword}' not allowed in code strings"
                    )

            namespace = {"inputs": copy.deepcopy(inputs), "output": None}
            namespace.update(safe_builtins)

            try:
                # nosec B102: exec with restricted namespace - dangerous operations filtered above
                exec(self.execution_logic, namespace)  # nosec B102
                result = namespace.get("output")
                if result is None:
                    raise RuntimeError("Code string did not set 'output' variable")
                return result
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute code string for principle {self.id}: {e}"
                )

        elif self.execution_type == "rule_spec":
            # Execute rule-based logic
            return self._execute_rules(inputs, self.execution_logic)
        else:
            raise ValueError(f"Unknown execution_type: {self.execution_type}")

    def _execute_rules(
        self, inputs: Dict[str, Any], rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute rule-based logic"""
        if not isinstance(rules, dict):
            raise ValueError("Rules must be a dictionary")

        output = {}

        for key, rule in rules.items():
            if isinstance(rule, dict):
                rule_type = rule.get("type", "direct")

                if rule_type == "direct":
                    # Direct mapping
                    source = rule.get("source")
                    if source in inputs:
                        output[key] = inputs[source]

                elif rule_type == "transform":
                    # Transform with function
                    source = rule.get("source")
                    transform = rule.get("transform")
                    if source in inputs and callable(transform):
                        try:
                            output[key] = transform(inputs[source])
                        except Exception as e:
                            raise RuntimeError(f"Transform failed for rule {key}: {e}")

                elif rule_type == "condition":
                    # Conditional logic
                    condition = rule.get("condition")
                    if_true = rule.get("if_true")
                    if_false = rule.get("if_false")
                    if callable(condition):
                        try:
                            result = condition(inputs)
                            output[key] = if_true if result else if_false
                        except Exception as e:
                            raise RuntimeError(
                                f"Condition evaluation failed for rule {key}: {e}"
                            )

        return output

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Principle):
            return False
        return self.id == other.id

    def __hash__(self):
        """Make principle hashable"""
        return hash(self.id)


@dataclass
class FailureAnalysis:
    """Analysis of validation failure"""

    failure_type: str
    root_cause: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    severity: float = 0.5
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "suggested_fixes": self.suggested_fixes,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Single validation result"""

    is_valid: bool
    confidence: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResults:
    """Results from multi-domain validation"""

    successful_domains: List[str] = field(default_factory=list)
    failed_domains: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    failure_analyses: Dict[str, FailureAnalysis] = field(default_factory=dict)
    domain_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.BASIC
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_success(self, domain: str, score: float = 1.0):
        """Add successful domain"""
        if not isinstance(domain, str):
            raise TypeError("Domain must be string")
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            raise ValueError("Score must be between 0 and 1")

        self.successful_domains.append(domain)
        self.domain_scores[domain] = score
        self._update_metrics()

    def add_failure(self, domain: str, analysis: FailureAnalysis):
        """Add failed domain"""
        if not isinstance(domain, str):
            raise TypeError("Domain must be string")
        if not isinstance(analysis, FailureAnalysis):
            raise TypeError("Analysis must be FailureAnalysis instance")

        self.failed_domains.append(domain)
        self.failure_analyses[domain] = analysis
        self.domain_scores[domain] = 0.0
        self._update_metrics()

    def _update_metrics(self):
        """Update success rate and confidence"""
        total = len(self.successful_domains) + len(self.failed_domains)
        if total > 0:
            self.success_rate = len(self.successful_domains) / total
            if self.domain_scores:
                self.overall_confidence = np.mean(list(self.domain_scores.values()))
            else:
                self.overall_confidence = 0.0
        else:
            self.success_rate = 0.0
            self.overall_confidence = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "successful_domains": self.successful_domains,
            "failed_domains": self.failed_domains,
            "success_rate": self.success_rate,
            "failure_analyses": {
                k: v.to_dict() for k, v in self.failure_analyses.items()
            },
            "domain_scores": self.domain_scores,
            "overall_confidence": self.overall_confidence,
            "validation_level": self.validation_level.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class DomainTestCase:
    """Test case for domain validation"""

    domain: str
    test_id: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "domain": self.domain,
            "test_id": self.test_id,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "constraints": self.constraints,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


class KnowledgeValidator:
    """Validates principle consistency and generalization"""

    def __init__(self, min_confidence: float = 0.6, consistency_threshold: float = 0.7):
        """
        Initialize knowledge validator

        Args:
            min_confidence: Minimum confidence threshold
            consistency_threshold: Threshold for consistency validation
        """
        self.min_confidence = min_confidence
        self.consistency_threshold = consistency_threshold

        # Domain validator
        self.domain_validator = DomainValidator()

        # Validation cache with expiry
        self.validation_cache = {}
        self.cache_timestamps = {}
        self.cache_size = 100
        self.cache_ttl = 3600  # 1 hour

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.total_validations = 0
        self.validation_history = deque(maxlen=1000)

        # Domain criticality map (safety-critical domains require more validation)
        self.domain_criticality = {
            "safety_critical": 0.95,
            "medical": 0.95,
            "financial": 0.90,
            "legal": 0.90,
            "control": 0.85,
            "autonomous_systems": 0.85,
            "security": 0.80,
            "planning": 0.75,
            "optimization": 0.70,
            "prediction": 0.65,
            "reasoning": 0.65,
            "classification": 0.60,
            "analysis": 0.55,
            "generation": 0.50,
            "perception": 0.50,
            "general": 0.30,
        }

        logger.info("KnowledgeValidator initialized")

    def _cleanup_expired_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired = []

        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired.append(key)

        for key in expired:
            if key in self.validation_cache:
                del self.validation_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]

    def validate(self, principle) -> ValidationResult:
        """
        Perform basic validation

        Args:
            principle: Principle to validate

        Returns:
            Validation result
        """
        try:
            errors = []
            warnings = []

            # Validate principle is correct type
            if not isinstance(principle, Principle):
                errors.append(f"Invalid principle type: {type(principle)}")
                return ValidationResult(is_valid=False, confidence=0.0, errors=errors)

            # Check basic requirements
            if not hasattr(principle, "id") or not principle.id:
                errors.append("Missing or empty principle ID")

            if not hasattr(principle, "confidence"):
                errors.append("Missing confidence score")
            elif not isinstance(principle.confidence, (int, float)):
                errors.append("Confidence must be numeric")
            elif principle.confidence < self.min_confidence:
                warnings.append(f"Low confidence: {principle.confidence:.2f}")

            if not hasattr(principle, "core_pattern"):
                errors.append("Missing core pattern")

            # Check domain specification
            if (
                not hasattr(principle, "applicable_domains")
                or not principle.applicable_domains
            ):
                warnings.append("No applicable domains specified")

            # Check execution logic
            if (
                not hasattr(principle, "execution_logic")
                or principle.execution_logic is None
            ):
                warnings.append(
                    "No execution logic defined - principle cannot be executed"
                )

            # Calculate validation confidence
            confidence = 1.0
            confidence -= len(errors) * 0.2
            confidence -= len(warnings) * 0.1
            confidence = max(0.0, min(1.0, confidence))

            result = ValidationResult(
                is_valid=len(errors) == 0,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metadata={"validation_type": "basic"},
            )

            self.total_validations += 1

            return result
        except Exception as e:
            logger.error("Error in basic validation: %s", e)
            return ValidationResult(
                is_valid=False, confidence=0.0, errors=[f"Validation error: {str(e)}"]
            )

    def validate_consistency(self, principle) -> ValidationResult:
        """
        Validate internal consistency of principle

        Args:
            principle: Principle to validate

        Returns:
            Validation result
        """
        with self.lock:
            try:
                # Cleanup expired cache entries
                self._cleanup_expired_cache()

                # Check cache
                cache_key = f"consistency_{getattr(principle, 'id', '')}"
                if cache_key in self.validation_cache:
                    cached_time = self.cache_timestamps.get(cache_key, 0)
                    if time.time() - cached_time < self.cache_ttl:
                        return self.validation_cache[cache_key]

                errors = []
                warnings = []
                consistency_scores = []

                # Check pattern consistency
                if hasattr(principle, "core_pattern"):
                    pattern_consistency = self._validate_pattern_consistency(
                        principle.core_pattern
                    )
                    consistency_scores.append(pattern_consistency)
                    if pattern_consistency < self.consistency_threshold:
                        warnings.append(
                            f"Pattern consistency below threshold: {pattern_consistency:.2f}"
                        )

                # Check domain consistency
                if hasattr(principle, "applicable_domains") and hasattr(
                    principle, "contraindicated_domains"
                ):
                    # Check for overlap
                    applicable_domains = getattr(principle, "applicable_domains", [])
                    contraindicated_domains = getattr(
                        principle, "contraindicated_domains", []
                    )
                    overlap = set(applicable_domains) & set(contraindicated_domains)
                    if overlap:
                        errors.append(f"Domain conflict: {overlap}")
                        consistency_scores.append(0.0)
                    else:
                        consistency_scores.append(1.0)

                # Check measurement requirements consistency
                if hasattr(principle, "measurement_requirements"):
                    req_consistency = self._validate_requirements_consistency(
                        getattr(principle, "measurement_requirements", [])
                    )
                    consistency_scores.append(req_consistency)
                    if req_consistency < self.consistency_threshold:
                        warnings.append("Inconsistent measurement requirements")

                # Check success/failure ratio
                if hasattr(principle, "success_count") and hasattr(
                    principle, "failure_count"
                ):
                    success_count = getattr(principle, "success_count", 0)
                    failure_count = getattr(principle, "failure_count", 0)
                    total = success_count + failure_count
                    if total > 0:
                        success_rate = success_count / total
                        consistency_scores.append(success_rate)
                        if success_rate < 0.5:
                            warnings.append(f"Low success rate: {success_rate:.2f}")

                # Calculate overall consistency
                if consistency_scores:
                    overall_consistency = np.mean(consistency_scores)
                else:
                    overall_consistency = 0.5

                result = ValidationResult(
                    is_valid=len(errors) == 0
                    and overall_consistency >= self.consistency_threshold,
                    confidence=overall_consistency,
                    errors=errors,
                    warnings=warnings,
                    metadata={
                        "validation_type": "consistency",
                        "consistency_scores": consistency_scores,
                    },
                )

                # Cache result
                if len(self.validation_cache) < self.cache_size:
                    self.validation_cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()

                # Track validation
                self.validation_history.append(
                    {
                        "principle_id": getattr(principle, "id", "unknown"),
                        "validation_type": "consistency",
                        "result": result.is_valid,
                        "confidence": result.confidence,
                        "timestamp": time.time(),
                    }
                )

                return result
            except Exception as e:
                logger.error("Error in consistency validation: %s", e)
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=[f"Consistency validation error: {str(e)}"],
                )

    def validate_across_domains(
        self, candidate, domains: List[str]
    ) -> ValidationResults:
        """
        Validate candidate across multiple domains

        Args:
            candidate: Candidate principle
            domains: List of domains to validate

        Returns:
            Multi-domain validation results
        """
        try:
            if not domains:
                return ValidationResults(
                    validation_level=ValidationLevel.DOMAIN_SPECIFIC
                )

            results = ValidationResults(
                validation_level=ValidationLevel.DOMAIN_SPECIFIC
            )

            for domain in domains:
                try:
                    # Generate domain-specific test
                    test_case = self.domain_validator.generate_domain_test(
                        candidate, domain
                    )

                    # Run test
                    test_result = self.domain_validator.run_sandboxed_test(
                        candidate, test_case
                    )

                    # Analyze result
                    if test_result.get("success", False):
                        results.add_success(domain, test_result.get("score", 1.0))
                    else:
                        # Analyze failure
                        failure = FailureAnalysis(
                            failure_type=test_result.get("failure_type", "unknown"),
                            error_message=test_result.get("error"),
                            root_cause=self._analyze_root_cause(test_result),
                            suggested_fixes=self._suggest_fixes(test_result, domain),
                            severity=test_result.get("severity", 0.5),
                        )
                        results.add_failure(domain, failure)
                except Exception as e:
                    logger.error("Error testing domain %s: %s", domain, e)
                    failure = FailureAnalysis(
                        failure_type="test_error", error_message=str(e), severity=0.8
                    )
                    results.add_failure(domain, failure)

            logger.info(
                "Cross-domain validation: %d/%d domains passed",
                len(results.successful_domains),
                len(domains),
            )

            return results
        except Exception as e:
            logger.error("Error in cross-domain validation: %s", e)
            return ValidationResults(validation_level=ValidationLevel.DOMAIN_SPECIFIC)

    def test_generalization(
        self, principle, test_domains: List[str] = None
    ) -> ValidationResults:
        """
        Test principle generalization across domains

        Args:
            principle: Principle to test
            test_domains: Domains to test generalization

        Returns:
            Generalization test results
        """
        try:
            results = ValidationResults(validation_level=ValidationLevel.GENERALIZATION)

            # Select diverse test domains if not provided
            if not test_domains:
                test_domains = self.domain_validator.select_diverse_domains(
                    principle, count=7
                )

            if not test_domains:
                logger.warning("No test domains available for generalization testing")
                return results

            # Test in each domain
            for domain in test_domains:
                try:
                    # Create generalization test
                    test_case = self._create_generalization_test(principle, domain)

                    # Run test
                    test_result = self.domain_validator.run_sandboxed_test(
                        principle, test_case
                    )

                    # Evaluate generalization
                    if self._passes_generalization_criteria(test_result, principle):
                        score = self._calculate_generalization_score(
                            test_result, principle
                        )
                        results.add_success(domain, score)
                    else:
                        failure = FailureAnalysis(
                            failure_type="generalization_failure",
                            error_message=f"Failed to generalize to {domain}",
                            root_cause=test_result.get("error"),
                            severity=0.6,
                            recoverable=True,
                            metadata={"domain": domain, "test_result": test_result},
                        )
                        results.add_failure(domain, failure)
                except Exception as e:
                    logger.error(
                        "Error testing generalization for domain %s: %s", domain, e
                    )
                    failure = FailureAnalysis(
                        failure_type="generalization_error",
                        error_message=str(e),
                        severity=0.7,
                    )
                    results.add_failure(domain, failure)

            # Calculate generalization metrics
            results.metadata["generalization_score"] = (
                self._calculate_overall_generalization(results)
            )

            logger.info(
                "Generalization test: %.2f success rate across %d domains",
                results.success_rate,
                len(test_domains),
            )

            return results
        except Exception as e:
            logger.error("Error in generalization testing: %s", e)
            return ValidationResults(validation_level=ValidationLevel.GENERALIZATION)

    def validate_principle_multilevel(
        self, principle, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Orchestrated multi-level validation - runs only necessary validation levels

        Args:
            principle: Principle to validate
            context: Optional context with budget/time constraints

        Returns:
            Dictionary of validation results by level
        """
        try:
            context = context or {}

            # EXAMINE: Select which validation levels to run
            levels_to_run = self._select_validation_levels(principle, context)

            logger.info(
                "Running validation levels: %s for principle %s",
                levels_to_run,
                getattr(principle, "id", "unknown"),
            )

            results = {}

            # APPLY: Run selected validations
            if ValidationLevel.BASIC in levels_to_run:
                results["basic"] = self.validate(principle)

                # Early exit if basic validation fails critically
                if not results["basic"].is_valid and results["basic"].confidence < 0.3:
                    logger.warning(
                        "Basic validation failed critically, skipping further validation"
                    )
                    results["skipped"] = ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        warnings=[
                            "Further validation skipped due to critical basic validation failure"
                        ],
                        metadata={
                            "levels_skipped": [
                                l.value
                                for l in levels_to_run
                                if l != ValidationLevel.BASIC
                            ]
                        },
                    )
                    return results

            if ValidationLevel.CONSISTENCY in levels_to_run:
                results["consistency"] = self.validate_consistency(principle)

            if ValidationLevel.DOMAIN_SPECIFIC in levels_to_run:
                # Get domains to test
                test_domains = self._get_domains_for_testing(principle, context)
                if test_domains:
                    results["domain_specific"] = self.validate_across_domains(
                        principle, test_domains
                    )
                else:
                    results["domain_specific"] = ValidationResult(
                        is_valid=True,
                        confidence=0.5,
                        warnings=["No domains available for testing"],
                        metadata={
                            "validation_type": "domain_specific",
                            "skipped": True,
                        },
                    )

            if ValidationLevel.GENERALIZATION in levels_to_run:
                # Get generalization test domains
                gen_domains = context.get("generalization_domains")
                if not gen_domains:
                    gen_domains = self.domain_validator.select_diverse_domains(
                        principle, count=7
                    )

                results["generalization"] = self.test_generalization(
                    principle, gen_domains
                )

            # REMEMBER: Track validation execution
            self.validation_history.append(
                {
                    "principle_id": getattr(principle, "id", "unknown"),
                    "validation_type": "multilevel",
                    "levels_run": [l.value for l in levels_to_run],
                    "results": {
                        k: v.is_valid
                        for k, v in results.items()
                        if hasattr(v, "is_valid")
                    },
                    "timestamp": time.time(),
                }
            )

            return results
        except Exception as e:
            logger.error("Error in multilevel validation: %s", e)
            return {
                "error": ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=[f"Multilevel validation error: {str(e)}"],
                )
            }

    def _select_validation_levels(
        self, principle, context: Dict[str, Any]
    ) -> List[ValidationLevel]:
        """
        Select which validation levels to run based on principle characteristics

        Args:
            principle: Principle to validate
            context: Context with constraints and hints

        Returns:
            List of validation levels to execute
        """
        levels = []

        # Get principle characteristics
        confidence = getattr(principle, "confidence", 0.5)
        domains = getattr(principle, "applicable_domains", ["general"])

        # Calculate domain criticality
        domain_criticality = self._get_domain_criticality(domains)

        # Get context constraints
        time_budget = context.get("time_budget_ms", float("inf"))
        quality_requirement = context.get("quality_requirement", "standard")
        force_comprehensive = context.get("force_comprehensive", False)

        # EXAMINE characteristics and SELECT levels

        # Always run basic validation
        levels.append(ValidationLevel.BASIC)

        # Run consistency if not extremely high confidence or if critical domain
        if confidence < 0.95 or domain_criticality > 0.8:
            levels.append(ValidationLevel.CONSISTENCY)

        # Domain-specific validation based on criticality
        if domain_criticality > 0.9:
            # Safety-critical: always validate across domains
            levels.append(ValidationLevel.DOMAIN_SPECIFIC)
        elif domain_criticality > 0.7:
            # Important domains: validate if enough time budget
            if time_budget > 1000:  # Need at least 1 second
                levels.append(ValidationLevel.DOMAIN_SPECIFIC)
        elif domain_criticality > 0.5 and confidence < 0.8:
            # Medium importance, lower confidence: validate to be safe
            if time_budget > 500:
                levels.append(ValidationLevel.DOMAIN_SPECIFIC)

        # Generalization testing
        if force_comprehensive or quality_requirement == "high":
            # Comprehensive validation requested
            levels.append(ValidationLevel.GENERALIZATION)
        elif domain_criticality > 0.85:
            # Critical domains need generalization testing
            levels.append(ValidationLevel.GENERALIZATION)
        elif confidence > 0.8 and len(domains) <= 2 and time_budget > 2000:
            # High confidence, few domains: test generalization
            levels.append(ValidationLevel.GENERALIZATION)

        # Special handling for low confidence principles
        if confidence < 0.6:
            # Low confidence: need thorough validation
            if ValidationLevel.DOMAIN_SPECIFIC not in levels:
                levels.append(ValidationLevel.DOMAIN_SPECIFIC)

        # Remove duplicates while preserving order
        seen = set()
        levels = [l for l in levels if not (l in seen or seen.add(l))]

        logger.debug(
            "Selected validation levels for principle (confidence=%.2f, criticality=%.2f): %s",
            confidence,
            domain_criticality,
            [l.value for l in levels],
        )

        return levels

    def _get_domain_criticality(self, domains: List[str]) -> float:
        """
        Calculate overall criticality score for a list of domains

        Args:
            domains: List of domain names

        Returns:
            Maximum criticality score (most critical domain)
        """
        if not domains:
            return 0.3  # Default to low criticality

        criticalities = []
        for domain in domains:
            # Normalize domain name (lowercase, strip)
            domain_normalized = str(domain).lower().strip()

            # Check exact match first
            if domain_normalized in self.domain_criticality:
                criticalities.append(self.domain_criticality[domain_normalized])
            else:
                # Check for partial matches (e.g., "safety_critical_control" contains "safety_critical")
                matched = False
                for known_domain, criticality in self.domain_criticality.items():
                    if (
                        known_domain in domain_normalized
                        or domain_normalized in known_domain
                    ):
                        criticalities.append(criticality)
                        matched = True
                        break

                if not matched:
                    # Unknown domain: use medium criticality
                    criticalities.append(0.5)

        # Return maximum criticality (most critical domain determines requirements)
        return max(criticalities) if criticalities else 0.3

    def _get_domains_for_testing(self, principle, context: Dict[str, Any]) -> List[str]:
        """
        Get list of domains to test for domain-specific validation

        Args:
            principle: Principle to test
            context: Context with hints

        Returns:
            List of domain names to test
        """
        # Check if domains specified in context
        if "test_domains" in context:
            return context["test_domains"]

        # Get principle's domains
        applicable_domains = getattr(principle, "applicable_domains", [])

        if not applicable_domains:
            # No domains specified, use general
            return ["general"]

        # For safety-critical domains, test all applicable domains
        criticality = self._get_domain_criticality(applicable_domains)
        if criticality > 0.85:
            return applicable_domains

        # For other domains, test up to 3-5 domains
        max_domains = context.get("max_test_domains", 3)
        if len(applicable_domains) <= max_domains:
            return applicable_domains

        # Select diverse subset
        return self.domain_validator.select_diverse_domains(
            principle, count=max_domains
        )

    def _validate_pattern_consistency(self, pattern) -> float:
        """Validate pattern internal consistency"""
        try:
            consistency = 1.0

            if hasattr(pattern, "components"):
                components = getattr(pattern, "components", [])
                # Check component consistency
                if not components:
                    consistency -= 0.3
                elif len(components) > 50:
                    consistency -= 0.2  # Too complex

            if hasattr(pattern, "confidence"):
                # Pattern confidence affects consistency
                pattern_conf = getattr(pattern, "confidence", 1.0)
                if isinstance(pattern_conf, (int, float)):
                    consistency *= pattern_conf

            return max(0.0, min(1.0, consistency))
        except Exception as e:
            logger.warning("Error validating pattern consistency: %s", e)
            return 0.5

    def _validate_requirements_consistency(self, requirements: List[str]) -> float:
        """Validate measurement requirements consistency"""
        try:
            if not requirements:
                return 1.0  # No requirements is consistent

            # Check for conflicting requirements
            conflicts = [
                ("maximize", "minimize"),
                ("increase", "decrease"),
                ("high", "low"),
            ]

            consistency = 1.0

            for req1 in requirements:
                for req2 in requirements:
                    if req1 != req2:
                        req1_lower = str(req1).lower()
                        req2_lower = str(req2).lower()
                        for conflict_pair in conflicts:
                            if (
                                conflict_pair[0] in req1_lower
                                and conflict_pair[1] in req2_lower
                            ) or (
                                conflict_pair[1] in req1_lower
                                and conflict_pair[0] in req2_lower
                            ):
                                consistency -= 0.2

            return max(0.0, min(1.0, consistency))
        except Exception as e:
            logger.warning("Error validating requirements consistency: %s", e)
            return 0.5

    def _analyze_root_cause(self, test_result: Dict[str, Any]) -> Optional[str]:
        """Analyze root cause of test failure"""
        try:
            if "error" in test_result:
                error = str(test_result["error"]).lower()

                # Pattern matching for common root causes
                if "timeout" in error:
                    return "execution_timeout"
                elif "memory" in error:
                    return "memory_exceeded"
                elif "type" in error:
                    return "type_mismatch"
                elif "domain" in error:
                    return "domain_incompatibility"
                else:
                    return "unknown_error"

            return None
        except Exception as e:
            return "analysis_error"

    def _suggest_fixes(self, test_result: Dict[str, Any], domain: str) -> List[str]:
        """Suggest fixes for test failure"""
        fixes = []

        try:
            root_cause = self._analyze_root_cause(test_result)

            if root_cause == "execution_timeout":
                fixes.append("Increase timeout or optimize execution")
            elif root_cause == "memory_exceeded":
                fixes.append("Reduce memory usage or increase allocation")
            elif root_cause == "domain_incompatibility":
                fixes.append(f"Add domain-specific adaptation for {domain}")
            elif root_cause == "type_mismatch":
                fixes.append("Verify input/output type compatibility")

            return fixes
        except Exception as e:
            logger.warning("Error suggesting fixes: %s", e)
            return ["Review principle implementation"]

    def _create_generalization_test(self, principle, domain: str) -> DomainTestCase:
        """Create generalization test case"""
        try:
            return DomainTestCase(
                domain=domain,
                test_id=f"gen_test_{domain}_{int(time.time())}",
                inputs={
                    "domain": domain,
                    "principle_id": getattr(principle, "id", "unknown"),
                    "test_type": "generalization",
                },
                expected_outputs=None,  # Generalization doesn't require exact outputs
                constraints={"allow_adaptation": True, "max_deviation": 0.3},
                timeout=60.0,
                metadata={"generalization_test": True},
            )
        except Exception as e:
            logger.error("Error creating generalization test: %s", e)
            raise

    def _passes_generalization_criteria(
        self, test_result: Dict[str, Any], principle
    ) -> bool:
        """Check if test passes generalization criteria"""
        try:
            if not test_result.get("success", False):
                return False

            # Check if core pattern was maintained
            if "pattern_match" in test_result:
                if test_result["pattern_match"] < 0.5:
                    return False

            # Check if performance is acceptable
            if "performance" in test_result:
                baseline_perf = getattr(principle, "baseline_performance", 0.5)
                if test_result["performance"] < baseline_perf * 0.7:
                    return False

            return True
        except Exception as e:
            logger.warning("Error checking generalization criteria: %s", e)
            return False

    def _calculate_generalization_score(
        self, test_result: Dict[str, Any], principle
    ) -> float:
        """Calculate generalization score"""
        try:
            score = 0.5  # Base score

            # Pattern preservation
            if "pattern_match" in test_result:
                score += test_result["pattern_match"] * 0.3

            # Performance retention
            if "performance" in test_result:
                baseline_perf = getattr(principle, "baseline_performance", 0.5)
                if baseline_perf > 0:
                    retention = test_result["performance"] / baseline_perf
                    score += min(1.0, retention) * 0.2

            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning("Error calculating generalization score: %s", e)
            return 0.5

    def _calculate_overall_generalization(self, results: ValidationResults) -> float:
        """Calculate overall generalization score"""
        try:
            if not results.domain_scores:
                return 0.0

            # Weight by domain diversity
            unique_domains = len(set(results.successful_domains))
            diversity_bonus = min(0.2, unique_domains * 0.02)

            # Base score from success rate
            base_score = results.success_rate

            # Average domain scores
            avg_score = np.mean(list(results.domain_scores.values()))

            return min(1.0, base_score * 0.5 + avg_score * 0.3 + diversity_bonus)
        except Exception as e:
            logger.warning("Error calculating overall generalization: %s", e)
            return 0.0


class DomainValidator:
    """Domain-specific validation"""

    def __init__(self):
        """Initialize domain validator"""
        self.domain_registry = self._initialize_domain_registry()
        self.test_cache = {}
        self.sandbox_config = {
            "timeout": 30,
            "memory_limit": 512 * 1024 * 1024,  # 512MB
            "cpu_limit": 1,
        }

        logger.info("DomainValidator initialized (platform: %s)", platform.system())

    def select_diverse_domains(self, candidate, count: int = 7) -> List[str]:
        """
        Select diverse domains for testing

        Args:
            candidate: Candidate principle
            count: Number of domains to select

        Returns:
            List of diverse domain names
        """
        try:
            all_domains = list(self.domain_registry.keys())

            if not all_domains:
                return []

            # Get candidate's origin domain
            origin_domain = getattr(candidate, "origin_domain", "general")
            if origin_domain is None:
                origin_domain = "general"

            # Categorize domains by similarity to origin
            domain_scores = {}
            for domain in all_domains:
                similarity = self._calculate_domain_similarity(origin_domain, domain)
                domain_scores[domain] = similarity

            # Sort by similarity
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1])

            # Select diverse set
            selected = []

            # Always include origin domain if available
            if origin_domain in all_domains:
                selected.append(origin_domain)

            # Add most similar domain
            if len(sorted_domains) > 0 and sorted_domains[-1][0] not in selected:
                selected.append(sorted_domains[-1][0])

            # Add least similar domain
            if len(sorted_domains) > 0 and sorted_domains[0][0] not in selected:
                selected.append(sorted_domains[0][0])

            # Add random domains from middle
            middle_domains = [
                d
                for d, _ in sorted_domains[
                    len(sorted_domains) // 4 : 3 * len(sorted_domains) // 4
                ]
            ]

            while len(selected) < count and middle_domains:
                domain = np.random.choice(middle_domains)
                if domain not in selected:
                    selected.append(domain)
                    middle_domains.remove(domain)

            # Fill remaining with any domains
            for domain in all_domains:
                if len(selected) >= count:
                    break
                if domain not in selected:
                    selected.append(domain)

            return selected[:count]
        except Exception as e:
            logger.error("Error selecting diverse domains: %s", e)
            return ["general"]

    def categorize_domains_by_data_availability(
        self,
    ) -> Dict[DomainCategory, List[str]]:
        """
        Categorize domains by data availability

        Returns:
            Dictionary mapping categories to domain lists
        """
        try:
            categorized = defaultdict(list)

            for domain, info in self.domain_registry.items():
                data_points = info.get("data_points", 0)

                if data_points >= 1000:
                    category = DomainCategory.HIGH_DATA
                elif data_points >= 100:
                    category = DomainCategory.MEDIUM_DATA
                elif data_points > 0:
                    category = DomainCategory.LOW_DATA
                else:
                    category = DomainCategory.NO_DATA

                categorized[category].append(domain)

            return dict(categorized)
        except Exception as e:
            logger.error("Error categorizing domains: %s", e)
            return {}

    def generate_domain_test(self, principle, domain: str) -> DomainTestCase:
        """
        Generate domain-specific test case

        Args:
            principle: Principle to test
            domain: Target domain

        Returns:
            Domain test case
        """
        try:
            # Check cache
            cache_key = f"{getattr(principle, 'id', '')}_{domain}"
            if cache_key in self.test_cache:
                return self.test_cache[cache_key]

            # Get domain info
            domain_info = self.domain_registry.get(domain, {})

            # Generate inputs based on domain
            inputs = self._generate_domain_inputs(domain, domain_info)

            # Generate expected outputs if principle can compute them
            expected = None
            if hasattr(principle, "compute_expected") and callable(
                principle.compute_expected
            ):
                try:
                    expected = principle.compute_expected(inputs)
                except Exception as e:
                    logger.debug(
                        "Could not compute expected outputs for principle %s: %s",
                        getattr(principle, "id", "unknown"),
                        e,
                    )
            elif "typical_outputs" in domain_info:
                expected = domain_info["typical_outputs"]

            # Set constraints
            constraints = {
                "domain": domain,
                "complexity": domain_info.get("complexity", "medium"),
                "data_availability": domain_info.get("data_points", 0),
            }

            test_case = DomainTestCase(
                domain=domain,
                test_id=f"test_{domain}_{hashlib.md5(cache_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                inputs=inputs,
                expected_outputs=expected,
                constraints=constraints,
                timeout=domain_info.get("typical_timeout", 30.0),
                metadata={"principle_id": getattr(principle, "id", "unknown")},
            )

            # Cache test case
            self.test_cache[cache_key] = test_case

            return test_case
        except Exception as e:
            logger.error("Error generating domain test: %s", e)
            raise

    def run_sandboxed_test(
        self, principle, test_case: DomainTestCase
    ) -> Dict[str, Any]:
        """
        Run test in sandboxed environment with REAL execution

        Args:
            principle: Principle to test
            test_case: Test case to run

        Returns:
            Test execution result
        """
        start_time = time.time()

        # Prepare test environment
        with self._create_sandbox() as sandbox_dir:
            try:
                # Create test script
                script_path = self._create_test_script(
                    principle, test_case, sandbox_dir
                )

                # Run test with resource limits based on platform
                if platform.system() == "Linux" and RESOURCE_AVAILABLE:
                    result = self._run_with_resource_limits(
                        script_path, test_case.timeout
                    )
                else:
                    result = self._run_with_subprocess_limits(
                        script_path, test_case.timeout
                    )

                # Parse results
                success = result.get("exit_code") == 0

                # Extract test metrics
                if success and result.get("output"):
                    try:
                        output_data = json.loads(result["output"])

                        test_result = {
                            "success": True,
                            "score": output_data.get("score", 1.0),
                            "performance": output_data.get("performance", {}),
                            "pattern_match": output_data.get("pattern_match", 1.0),
                            "output": output_data.get("output"),
                            "execution_time": time.time() - start_time,
                        }
                    except json.JSONDecodeError as e:
                        test_result = {
                            "success": False,
                            "error": f"Failed to parse test output: {e}",
                            "failure_type": "parse_error",
                            "execution_time": time.time() - start_time,
                        }
                else:
                    test_result = {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "failure_type": "execution_failure",
                        "execution_time": time.time() - start_time,
                    }

            except Exception as e:
                test_result = {
                    "success": False,
                    "error": str(e),
                    "failure_type": "setup_failure",
                    "execution_time": time.time() - start_time,
                }

        logger.debug(
            "Sandboxed test for domain %s: %s",
            test_case.domain,
            "passed" if test_result.get("success") else "failed",
        )

        return test_result

    def _initialize_domain_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain registry with metadata"""
        return {
            "general": {
                "data_points": 10000,
                "complexity": "medium",
                "typical_timeout": 30,
                "characteristics": ["broad_applicability", "moderate_complexity"],
            },
            "optimization": {
                "data_points": 5000,
                "complexity": "high",
                "typical_timeout": 60,
                "characteristics": ["iterative", "metric_driven"],
            },
            "classification": {
                "data_points": 8000,
                "complexity": "medium",
                "typical_timeout": 20,
                "characteristics": ["categorical", "accuracy_focused"],
            },
            "prediction": {
                "data_points": 3000,
                "complexity": "high",
                "typical_timeout": 45,
                "characteristics": ["temporal", "uncertainty"],
            },
            "generation": {
                "data_points": 2000,
                "complexity": "high",
                "typical_timeout": 90,
                "characteristics": ["creative", "quality_subjective"],
            },
            "analysis": {
                "data_points": 6000,
                "complexity": "medium",
                "typical_timeout": 40,
                "characteristics": ["exploratory", "insight_driven"],
            },
            "control": {
                "data_points": 1500,
                "complexity": "high",
                "typical_timeout": 50,
                "characteristics": ["real_time", "stability_critical"],
            },
            "planning": {
                "data_points": 1000,
                "complexity": "very_high",
                "typical_timeout": 120,
                "characteristics": ["sequential", "constraint_heavy"],
            },
            "reasoning": {
                "data_points": 4000,
                "complexity": "very_high",
                "typical_timeout": 60,
                "characteristics": ["logical", "inference_based"],
            },
            "perception": {
                "data_points": 7000,
                "complexity": "medium",
                "typical_timeout": 25,
                "characteristics": ["sensory", "pattern_recognition"],
            },
        }

    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains"""
        try:
            if domain1 == domain2:
                return 1.0

            # Get domain characteristics
            info1 = self.domain_registry.get(domain1, {})
            info2 = self.domain_registry.get(domain2, {})

            chars1 = set(info1.get("characteristics", []))
            chars2 = set(info2.get("characteristics", []))

            if not chars1 or not chars2:
                return 0.5

            # Jaccard similarity
            intersection = len(chars1 & chars2)
            union = len(chars1 | chars2)

            if union == 0:
                return 0.0

            return intersection / union
        except Exception as e:
            logger.warning("Error calculating domain similarity: %s", e)
            return 0.5

    def _generate_domain_inputs(
        self, domain: str, domain_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate domain-specific inputs"""
        try:
            inputs = {"domain": domain, "test_type": "domain_validation"}

            # Add domain-specific inputs
            if domain == "optimization":
                inputs["objective"] = "minimize"
                inputs["constraints"] = []
                inputs["variables"] = ["x", "y"]
            elif domain == "classification":
                inputs["classes"] = ["A", "B", "C"]
                inputs["features"] = np.random.randn(10, 5).tolist()
            elif domain == "prediction":
                inputs["time_series"] = np.random.randn(20).tolist()
                inputs["horizon"] = 5
            elif domain == "generation":
                inputs["seed"] = int(np.random.randint(1000))
                inputs["length"] = 100
            elif domain == "analysis":
                inputs["data"] = np.random.randn(50, 10).tolist()
                inputs["analysis_type"] = "exploratory"

            return inputs
        except Exception as e:
            logger.error("Error generating domain inputs: %s", e)
            return {"domain": domain, "test_type": "domain_validation"}

    @contextmanager
    def _create_sandbox(self):
        """Create sandboxed test environment"""
        sandbox_dir = tempfile.mkdtemp(prefix="validator_sandbox_")

        try:
            yield sandbox_dir
        finally:
            # Cleanup
            import shutil

            try:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception as e:
                pass  # Best effort cleanup

    def _create_test_script(
        self, principle, test_case: DomainTestCase, sandbox_dir: str
    ) -> str:
        """
        Create test script for REAL execution

        Extracts and executes the principle's logic against test inputs
        """
        try:
            script_path = Path(sandbox_dir) / "test.py"

            # Extract principle data
            principle_id = getattr(principle, "id", "unknown")
            execution_type = getattr(principle, "execution_type", "function")

            # Serialize execution logic based on type
            if execution_type == "function" and hasattr(principle, "execution_logic"):
                execution_logic = principle.execution_logic
                if callable(execution_logic):
                    try:
                        logic_source = inspect.getsource(execution_logic)
                        logic_code = logic_source
                    except (OSError, TypeError):
                        # Can't extract source - try cloudpickle
                        try:
                            import cloudpickle

                            pickled = cloudpickle.dumps(execution_logic)
                            logic_code = f"""
import cloudpickle
execution_logic = cloudpickle.loads({pickled!r})
"""
                        except ImportError:
                            # Cloudpickle not available
                            logic_code = """
def execution_logic(inputs):
    raise NotImplementedError("Cannot serialize this function type (cloudpickle not available)")
"""
                        except Exception as e:
                            logic_code = f"""
def execution_logic(inputs):
    raise NotImplementedError("Cannot serialize function: {str(e)}")
"""
                else:
                    logic_code = """
def execution_logic(inputs):
    raise NotImplementedError("execution_logic is not callable")
"""
            elif execution_type == "code_string" and hasattr(
                principle, "execution_logic"
            ):
                # Code string is already executable
                logic_code = str(principle.execution_logic)
                if "def execution_logic" not in logic_code:
                    # Wrap code in function
                    logic_code = f"""
def execution_logic(inputs):
    {logic_code}
    return output
"""
            else:
                # No execution logic defined
                logic_code = """
def execution_logic(inputs):
    raise NotImplementedError("Principle has no execution logic defined")
"""

            # Generate complete test script
            script = f"""
import json
import sys
import traceback
import time

# Test case: {test_case.test_id}
# Principle: {principle_id}
# Domain: {test_case.domain}

# Principle execution logic
{logic_code}

def calculate_match_score(output, expected):
    \"\"\"Calculate how well output matches expected\"\"\"
    if expected is None:
        # No expected output - just verify execution succeeded
        return 1.0

    if output == expected:
        return 1.0

    # For dict outputs, calculate partial match
    if isinstance(output, dict) and isinstance(expected, dict):
        if not expected:
            return 1.0
        matches = sum(1 for k in expected if output.get(k) == expected[k])
        return matches / len(expected)

    # For list outputs, calculate element-wise match
    if isinstance(output, list) and isinstance(expected, list):
        if not expected:
            return 1.0
        if len(output) != len(expected):
            return 0.0
        matches = sum(1 for o, e in zip(output, expected) if o == e)
        return matches / len(expected)

    # For numeric outputs, calculate proximity
    try:
        output_num = float(output)
        expected_num = float(expected)
        diff = abs(output_num - expected_num)
        max_val = max(abs(output_num), abs(expected_num), 1.0)
        return max(0.0, 1.0 - (diff / max_val))
    except (ValueError, TypeError):
        pass

    # Default: no match
    return 0.0

def run_test():
    inputs = {json.dumps(test_case.inputs)}
    expected = {json.dumps(test_case.expected_outputs)}

    start_time = time.time()

    try:
        # EXECUTE THE PRINCIPLE
        output = execution_logic(inputs)

        execution_time = time.time() - start_time

        # Calculate score
        score = calculate_match_score(output, expected)

        # Pattern match is same as score for now
        pattern_match = score

        result = {{
            'score': score,
            'output': output,
            'performance': {{
                'time': execution_time,
                'success': True
            }},
            'pattern_match': pattern_match
        }}

    except Exception as e:
        execution_time = time.time() - start_time

        result = {{
            'error': str(e),
            'traceback': traceback.format_exc(),
            'score': 0.0,
            'performance': {{
                'time': execution_time,
                'success': False
            }},
            'pattern_match': 0.0
        }}

    return result

if __name__ == '__main__':
    try:
        result = run_test()
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        error_result = {{
            'error': str(e),
            'traceback': traceback.format_exc(),
            'score': 0.0
        }}
        print(json.dumps(error_result))
        sys.exit(1)
"""

            script_path.write_text(script)

            return str(script_path)
        except Exception as e:
            logger.error("Error creating test script: %s", e)
            raise

    def _run_with_resource_limits(
        self, script_path: str, timeout: float
    ) -> Dict[str, Any]:
        """Run script with resource limits (Linux only)"""
        try:
            if not RESOURCE_AVAILABLE:
                return self._run_with_subprocess_limits(script_path, timeout)

            python_exe = sys.executable or "python"

            def set_limits():
                """Set resource limits for child process"""
                try:
                    # CPU time limit
                    resource.setrlimit(
                        resource.RLIMIT_CPU, (int(timeout), int(timeout) + 5)
                    )
                    # Memory limit
                    resource.setrlimit(
                        resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)
                    )
                except Exception as e:
                    logger.warning("Could not set resource limits: %s", e)

            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=set_limits,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                return {
                    "exit_code": process.returncode,
                    "output": stdout,
                    "error": stderr if stderr else None,
                }
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.communicate(timeout=1)
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )
                return {"exit_code": -1, "output": None, "error": "Timeout exceeded"}
        except Exception as e:
            logger.error("Error running with resource limits: %s", e)
            return {"exit_code": -1, "output": None, "error": str(e)}

    def _run_with_subprocess_limits(
        self, script_path: str, timeout: float
    ) -> Dict[str, Any]:
        """Run script with subprocess limits (cross-platform)"""
        try:
            python_exe = sys.executable or "python"

            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                return {
                    "exit_code": process.returncode,
                    "output": stdout,
                    "error": stderr if stderr else None,
                }
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.communicate(timeout=1)
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )
                return {"exit_code": -1, "output": None, "error": "Timeout exceeded"}
        except Exception as e:
            logger.error("Error running subprocess: %s", e)
            return {"exit_code": -1, "output": None, "error": str(e)}


class ImbalanceHandler:
    """Handles validation imbalances across domains"""

    def __init__(self, imbalance_threshold: float = 0.3):
        """
        Initialize imbalance handler

        Args:
            imbalance_threshold: Threshold for detecting imbalance
        """
        self.imbalance_threshold = imbalance_threshold
        self.retest_history = deque(maxlen=100)

        logger.info("ImbalanceHandler initialized")

    def detect_imbalance(
        self, validation_results: ValidationResults
    ) -> Dict[str, float]:
        """
        Detect imbalances in validation results

        Args:
            validation_results: Validation results to analyze

        Returns:
            Dictionary of imbalance scores by category
        """
        try:
            imbalances = {}

            # Check success rate imbalance
            if validation_results.success_rate < 0.3:
                imbalances["low_success"] = 1.0 - validation_results.success_rate
            elif validation_results.success_rate > 0.95:
                imbalances["too_easy"] = validation_results.success_rate - 0.95

            # Check domain score variance
            if validation_results.domain_scores:
                scores = list(validation_results.domain_scores.values())
                if len(scores) > 1:
                    variance = np.var(scores)
                    if variance > self.imbalance_threshold:
                        imbalances["high_variance"] = variance

            # Check failure clustering
            if validation_results.failure_analyses:
                failure_types = [
                    fa.failure_type
                    for fa in validation_results.failure_analyses.values()
                ]
                if failure_types:
                    type_counts = Counter(failure_types)
                    most_common = type_counts.most_common(1)[0]
                    if most_common[1] > len(failure_types) * 0.7:
                        imbalances["failure_clustering"] = most_common[1] / len(
                            failure_types
                        )

            return imbalances
        except Exception as e:
            logger.error("Error detecting imbalance: %s", e)
            return {}

    def handle_imbalanced_validation(
        self, results: ValidationResults, domain: str
    ) -> ValidationResult:
        """
        Handle imbalanced validation for specific domain

        Args:
            results: Current validation results
            domain: Domain with imbalance

        Returns:
            Adjusted validation result
        """
        try:
            # Detect specific imbalance type
            imbalances = self.detect_imbalance(results)

            if "low_success" in imbalances:
                # Too difficult - adjust criteria
                return self._handle_low_success(results, domain)
            elif "too_easy" in imbalances:
                # Too easy - increase difficulty
                return self._handle_high_success(results, domain)
            elif "high_variance" in imbalances:
                # Inconsistent - need more testing
                return self._handle_high_variance(results, domain)
            else:
                # No significant imbalance
                return ValidationResult(
                    is_valid=domain in results.successful_domains,
                    confidence=results.domain_scores.get(domain, 0.0),
                )
        except Exception as e:
            logger.error("Error handling imbalanced validation: %s", e)
            return ValidationResult(is_valid=False, confidence=0.0, errors=[str(e)])

    def generate_diverse_test_set(self, domain: str) -> List[DomainTestCase]:
        """
        Generate diverse test set for domain

        Args:
            domain: Target domain

        Returns:
            List of diverse test cases
        """
        try:
            test_cases = []

            # Generate tests with different characteristics
            characteristics = [
                {"complexity": "low", "size": "small"},
                {"complexity": "low", "size": "large"},
                {"complexity": "high", "size": "small"},
                {"complexity": "high", "size": "large"},
                {"complexity": "medium", "size": "medium"},
            ]

            for i, chars in enumerate(characteristics):
                test_case = DomainTestCase(
                    domain=domain,
                    test_id=f"diverse_{domain}_{i}",
                    inputs=self._generate_test_inputs(chars),
                    constraints=chars,
                    timeout=30.0 * (2.0 if chars["complexity"] == "high" else 1.0),
                )
                test_cases.append(test_case)

            return test_cases
        except Exception as e:
            logger.error("Error generating diverse test set: %s", e)
            return []

    def retest_with_stratification(self, domain: str, principle) -> ValidationResult:
        """
        Retest with stratified sampling

        Args:
            domain: Domain to retest
            principle: Principle to validate

        Returns:
            Stratified validation result
        """
        try:
            # Generate stratified test set
            test_cases = self.generate_diverse_test_set(domain)

            if not test_cases:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=["Could not generate test cases"],
                )

            # Run tests
            results = []
            validator = DomainValidator()
            for test_case in test_cases:
                test_result = validator.run_sandboxed_test(principle, test_case)
                results.append(test_result)

            # Aggregate results
            successes = sum(1 for r in results if r.get("success", False))
            scores = [r.get("score", 0.0) for r in results if r.get("success", False)]

            # Track retest
            self.retest_history.append(
                {
                    "domain": domain,
                    "principle_id": getattr(principle, "id", "unknown"),
                    "test_count": len(test_cases),
                    "success_count": successes,
                    "timestamp": time.time(),
                }
            )

            return ValidationResult(
                is_valid=successes > len(test_cases) / 2,
                confidence=np.mean(scores) if scores else 0.0,
                metadata={"stratified": True, "test_count": len(test_cases)},
            )
        except Exception as e:
            logger.error("Error in stratified retest: %s", e)
            return ValidationResult(is_valid=False, confidence=0.0, errors=[str(e)])

    def _handle_low_success(
        self, results: ValidationResults, domain: str
    ) -> ValidationResult:
        """Handle low success rate"""
        try:
            # Relax validation criteria
            adjusted_confidence = results.domain_scores.get(domain, 0.0) * 1.5
            adjusted_confidence = min(
                0.7, adjusted_confidence
            )  # Cap at moderate confidence

            return ValidationResult(
                is_valid=adjusted_confidence > 0.3,
                confidence=adjusted_confidence,
                warnings=["Validation criteria relaxed due to low success rate"],
                metadata={"adjusted": True, "reason": "low_success"},
            )
        except Exception as e:
            logger.error("Error handling low success: %s", e)
            return ValidationResult(is_valid=False, confidence=0.0, errors=[str(e)])

    def _handle_high_success(
        self, results: ValidationResults, domain: str
    ) -> ValidationResult:
        """Handle very high success rate"""
        try:
            # Increase scrutiny
            adjusted_confidence = results.domain_scores.get(domain, 1.0) * 0.8

            return ValidationResult(
                is_valid=adjusted_confidence > 0.7,
                confidence=adjusted_confidence,
                warnings=["Validation may be too lenient"],
                metadata={"adjusted": True, "reason": "high_success"},
            )
        except Exception as e:
            logger.error("Error handling high success: %s", e)
            return ValidationResult(is_valid=False, confidence=0.0, errors=[str(e)])

    def _handle_high_variance(
        self, results: ValidationResults, domain: str
    ) -> ValidationResult:
        """Handle high variance in results"""
        try:
            # Use median instead of mean
            scores = list(results.domain_scores.values())
            median_score = np.median(scores) if scores else 0.5

            return ValidationResult(
                is_valid=median_score > 0.5,
                confidence=median_score,
                warnings=["High variance in validation results"],
                metadata={
                    "adjusted": True,
                    "reason": "high_variance",
                    "used_median": True,
                },
            )
        except Exception as e:
            logger.error("Error handling high variance: %s", e)
            return ValidationResult(is_valid=False, confidence=0.0, errors=[str(e)])

    def _generate_test_inputs(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test inputs based on characteristics"""
        try:
            size_map = {"small": 10, "medium": 100, "large": 1000}
            complexity_map = {"low": 1, "medium": 5, "high": 10}

            size = size_map.get(characteristics.get("size", "medium"), 100)
            complexity = complexity_map.get(
                characteristics.get("complexity", "medium"), 5
            )

            return {
                "data_size": size,
                "feature_count": complexity,
                "data": np.random.randn(min(size, 100), complexity).tolist(),
                "characteristics": characteristics,
            }
        except Exception as e:
            logger.error("Error generating test inputs: %s", e)
            return {"characteristics": characteristics}
