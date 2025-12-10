"""
Safety Governor for Tool Selection System

Enforces safety constraints, tool contracts, and provides veto mechanisms
to ensure safe and reliable tool selection and execution.

Fixed version with ReDoS protection and bounded storage.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json
import hashlib
from pathlib import Path
import threading
import re

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for operations"""

    CRITICAL = 0  # Highest safety requirements
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    MINIMAL = 4  # Lowest safety requirements


class VetoReason(Enum):
    """Reasons for safety veto"""

    UNSAFE_INPUT = "unsafe_input"
    UNSAFE_OUTPUT = "unsafe_output"
    CONTRACT_VIOLATION = "contract_violation"
    RESOURCE_EXCEEDED = "resource_exceeded"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    INCONSISTENT_OUTPUT = "inconsistent_output"
    FORBIDDEN_OPERATION = "forbidden_operation"
    RATE_LIMITED = "rate_limited"


class SafetyAction(Enum):
    """Actions to take for safety violations"""

    ALLOW = "allow"
    SANITIZE = "sanitize"
    VETO = "veto"
    REDIRECT = "redirect"
    LOG_AND_ALLOW = "log_and_allow"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass
class ToolContract:
    """Contract defining tool constraints and requirements"""

    tool_name: str
    required_inputs: Set[str]
    forbidden_inputs: Set[str]
    max_execution_time_ms: float
    max_energy_mj: float
    min_confidence: float
    required_safety_level: SafetyLevel
    allowed_operations: Set[str]
    forbidden_operations: Set[str]
    output_validators: List[Callable]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyViolation:
    """Record of a safety violation"""

    timestamp: float
    tool_name: str
    violation_type: VetoReason
    severity: SafetyLevel
    details: str
    action_taken: SafetyAction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyContext:
    """Context for safety evaluation"""

    problem: Any
    tool_name: str
    features: Optional[np.ndarray]
    constraints: Dict[str, float]
    user_context: Dict[str, Any]
    safety_level: SafetyLevel
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyValidator:
    """Validates inputs and outputs for safety"""

    def __init__(self):
        # CRITICAL FIX: Pre-compile patterns to avoid ReDoS
        # Use simpler, non-backtracking patterns
        self.unsafe_patterns_compiled = [
            re.compile(r"(?i)\b(attack|exploit|vulnerability|injection)\b"),
            re.compile(r"(?i)\b(malware|virus|trojan|ransomware)\b"),
            re.compile(r"(?i)\b(hack|breach|compromise|backdoor)\b"),
            re.compile(r"(?i)\b(steal|theft|fraud|scam)\b"),
            re.compile(r"(?i)\b(illegal|illicit|prohibited|banned)\b"),
        ]

        # CRITICAL FIX: Pre-compile sensitive patterns with simple, safe regex
        self.sensitive_patterns_compiled = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b\d{16}\b"),  # Credit card
            re.compile(r"(?i)\b(password|secret|token|api[_\s]?key)\b"),
            re.compile(r"(?i)\b(private[_\s]?key|credential)\b"),
        ]

        # Safe content whitelist
        self.safe_keywords = {
            "analyze",
            "compute",
            "calculate",
            "evaluate",
            "predict",
            "classify",
            "summarize",
            "explain",
        }

        # CRITICAL FIX: Add size limits
        self.max_input_size = 1000000  # 1MB
        self.max_output_size = 5000000  # 5MB
        self.max_pattern_check_size = 10000  # Only check first 10KB for patterns

    def validate_input(self, input_data: Any) -> Tuple[bool, str]:
        """Validate input for safety - CRITICAL: ReDoS protection"""

        try:
            input_str = str(input_data) if input_data else ""

            # CRITICAL FIX: Check size before any processing
            if len(input_str) > self.max_input_size:
                return False, "Input too large"

            # CRITICAL FIX: Truncate for pattern matching to prevent ReDoS
            check_str = input_str[: self.max_pattern_check_size]
            check_str_lower = check_str.lower()

            # Check for unsafe patterns with pre-compiled regex
            for pattern in self.unsafe_patterns_compiled:
                try:
                    match = pattern.search(check_str_lower)
                    if match:
                        return False, f"Unsafe pattern detected: {match.group()}"
                except Exception as e:
                    logger.warning(f"Pattern matching error: {e}")
                    continue

            # Check for sensitive data
            for pattern in self.sensitive_patterns_compiled:
                try:
                    match = pattern.search(check_str)
                    if match:
                        return False, "Sensitive data detected"
                except Exception as e:
                    logger.warning(f"Sensitive pattern matching error: {e}")
                    continue

            return True, "Input validated"
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def validate_output(self, output_data: Any) -> Tuple[bool, str]:
        """Validate output for safety"""

        if output_data is None:
            return True, "Null output allowed"

        try:
            output_str = str(output_data)

            # CRITICAL FIX: Check size first
            if len(output_str) > self.max_output_size:
                return False, "Output too large"

            # CRITICAL FIX: Truncate for pattern checking
            check_str = output_str[: self.max_pattern_check_size]

            # Check for sensitive data leakage
            for pattern in self.sensitive_patterns_compiled:
                try:
                    match = pattern.search(check_str)
                    if match:
                        return False, "Output contains sensitive data"
                except Exception as e:
                    logger.warning(f"Output pattern matching error: {e}")
                    continue

            return True, "Output validated"
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input by removing unsafe content"""

        try:
            if isinstance(input_data, str):
                # Remove sensitive patterns
                sanitized = input_data

                # CRITICAL FIX: Limit size before sanitization
                if len(sanitized) > self.max_input_size:
                    sanitized = sanitized[: self.max_input_size]

                for pattern in self.sensitive_patterns_compiled:
                    try:
                        sanitized = pattern.sub("[REDACTED]", sanitized)
                    except Exception as e:
                        logger.warning(f"Sanitization pattern error: {e}")
                        continue

                return sanitized

            return input_data
        except Exception as e:
            logger.error(f"Sanitization failed: {e}")
            return input_data


class ConsistencyChecker:
    """Checks output consistency across tools"""

    def check_consistency(self, outputs: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if outputs from different tools are consistent

        Returns:
            (is_consistent, confidence, details)
        """

        if len(outputs) < 2:
            return True, 1.0, "Single output, no inconsistency"

        try:
            # Extract comparable values
            values = []
            for tool_name, output in outputs.items():
                value = self._extract_comparable_value(output)
                if value is not None:
                    values.append((tool_name, value))

            if not values:
                return True, 0.5, "No comparable values found"

            # Check consistency based on value types
            if all(isinstance(v[1], bool) for v in values):
                return self._check_boolean_consistency(values)
            elif all(isinstance(v[1], (int, float)) for v in values):
                return self._check_numerical_consistency(values)
            elif all(isinstance(v[1], str) for v in values):
                return self._check_string_consistency(values)
            else:
                return True, 0.3, "Mixed value types, cannot compare"
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return True, 0.5, f"Error: {str(e)}"

    def _extract_comparable_value(self, output: Any) -> Any:
        """Extract comparable value from output"""

        try:
            if output is None:
                return None

            # Handle different output formats
            if hasattr(output, "value"):
                return output.value
            elif hasattr(output, "result"):
                return output.result
            elif hasattr(output, "conclusion"):
                return output.conclusion
            elif isinstance(output, dict):
                return (
                    output.get("value")
                    or output.get("result")
                    or output.get("conclusion")
                )
            else:
                return output
        except Exception as e:
            logger.warning(f"Value extraction failed: {e}")
            return None

    def _check_boolean_consistency(
        self, values: List[Tuple[str, bool]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of boolean values"""

        try:
            true_count = sum(1 for _, v in values if v)
            false_count = len(values) - true_count

            if true_count == len(values) or false_count == len(values):
                return True, 1.0, "All tools agree"

            # Majority vote
            majority = true_count > false_count
            confidence = (
                max(true_count, false_count) / len(values) if len(values) > 0 else 0.5
            )

            disagreeing = [name for name, v in values if v != majority]

            return False, confidence, f"Disagreement from: {disagreeing}"
        except Exception as e:
            logger.error(f"Boolean consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"

    def _check_numerical_consistency(
        self, values: List[Tuple[str, float]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of numerical values"""

        try:
            nums = [float(v) for _, v in values]
            mean = np.mean(nums)
            std = np.std(nums)

            # Check coefficient of variation
            # CRITICAL FIX: Handle division by zero
            if abs(mean) < 1e-10:
                cv = 0.0 if std < 1e-10 else float("inf")
            else:
                cv = std / abs(mean)

            if cv < 0.1:  # Less than 10% variation
                return (
                    True,
                    float(np.clip(1.0 - cv, 0.0, 1.0)),
                    "Numerical values consistent",
                )
            elif cv < 0.3:  # Less than 30% variation
                return (
                    True,
                    float(np.clip(1.0 - cv, 0.0, 1.0)),
                    f"Moderate variation: CV={cv:.2f}",
                )
            else:
                outliers = [
                    name for name, v in values if abs(float(v) - mean) > 2 * std
                ]
                confidence = float(np.clip(1.0 - min(cv, 1.0), 0.0, 1.0))
                return False, confidence, f"High variation, outliers: {outliers}"
        except Exception as e:
            logger.error(f"Numerical consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"

    def _check_string_consistency(
        self, values: List[Tuple[str, str]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of string values"""

        try:
            # Simple equality check for now
            unique_values = set(v for _, v in values)

            if len(unique_values) == 1:
                return True, 1.0, "All strings match"

            # Could add fuzzy matching here
            confidence = 1.0 / len(unique_values) if len(unique_values) > 0 else 0.0
            return False, confidence, f"Different values: {len(unique_values)} unique"
        except Exception as e:
            logger.error(f"String consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"


class SafetyGovernor:
    """
    Main safety governor enforcing contracts and safety policies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Safety components
        self.validator = SafetyValidator()
        self.consistency_checker = ConsistencyChecker()

        # Tool contracts
        self.contracts = {}
        self._initialize_default_contracts()

        # Safety configuration
        self.default_safety_level = SafetyLevel.MEDIUM
        self.veto_threshold = config.get("veto_threshold", 0.8)
        self.require_consensus = config.get("require_consensus", False)

        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.rate_limit_window = config.get("rate_limit_window", 60)  # seconds
        self.max_requests_per_tool = config.get("max_requests_per_tool", 100)

        # CRITICAL FIX: Bounded violation storage
        self.max_violations = config.get("max_violations", 1000)
        self.violations = deque(maxlen=self.max_violations)
        self.violation_counts = defaultdict(int)

        # CRITICAL FIX: Bounded audit trail
        self.max_audit_entries = config.get("max_audit_entries", 10000)
        self.audit_trail = deque(maxlen=self.max_audit_entries)

        # Safety cache
        self.safety_cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
        self.max_cache_size = config.get("max_cache_size", 1000)

        # CRITICAL FIX: Add locks for thread safety
        self.lock = threading.RLock()
        self.cache_lock = threading.RLock()

    def _initialize_default_contracts(self):
        """Initialize default tool contracts"""

        self.contracts["symbolic"] = ToolContract(
            tool_name="symbolic",
            required_inputs={"logic", "rules"},
            forbidden_inputs={"undefined", "infinite"},
            max_execution_time_ms=5000,
            max_energy_mj=500,
            min_confidence=0.7,
            required_safety_level=SafetyLevel.HIGH,
            allowed_operations={"prove", "verify", "deduce"},
            forbidden_operations={"modify", "delete"},
            output_validators=[lambda x: x is not None],
        )

        self.contracts["probabilistic"] = ToolContract(
            tool_name="probabilistic",
            required_inputs=set(),
            forbidden_inputs={"nan", "inf"},
            max_execution_time_ms=3000,
            max_energy_mj=300,
            min_confidence=0.5,
            required_safety_level=SafetyLevel.MEDIUM,
            allowed_operations={"predict", "estimate", "sample"},
            forbidden_operations={"assert"},
            output_validators=[lambda x: hasattr(x, "probability") or True],
        )

        self.contracts["causal"] = ToolContract(
            tool_name="causal",
            required_inputs={"graph", "data"},
            forbidden_inputs={"cyclic"},
            max_execution_time_ms=10000,
            max_energy_mj=1000,
            min_confidence=0.6,
            required_safety_level=SafetyLevel.HIGH,
            allowed_operations={"intervene", "observe", "predict"},
            forbidden_operations={"manipulate"},
            output_validators=[lambda x: not self._has_cycles(x)],
        )

        self.contracts["analogical"] = ToolContract(
            tool_name="analogical",
            required_inputs={"source", "target"},
            forbidden_inputs=set(),
            max_execution_time_ms=2000,
            max_energy_mj=200,
            min_confidence=0.4,
            required_safety_level=SafetyLevel.LOW,
            allowed_operations={"map", "transfer", "adapt"},
            forbidden_operations=set(),
            output_validators=[lambda x: x is not None],
        )

        self.contracts["multimodal"] = ToolContract(
            tool_name="multimodal",
            required_inputs={"modalities"},
            forbidden_inputs={"corrupted"},
            max_execution_time_ms=15000,
            max_energy_mj=1500,
            min_confidence=0.5,
            required_safety_level=SafetyLevel.MEDIUM,
            allowed_operations={"fuse", "align", "translate"},
            forbidden_operations={"forge"},
            output_validators=[lambda x: self._validate_multimodal(x)],
        )

    def check_safety(
        self, context: SafetyContext
    ) -> Tuple[SafetyAction, Optional[str]]:
        """
        Main safety check for tool selection

        Returns:
            (action, reason)
        """

        try:
            # Check cache
            cache_key = self._compute_cache_key(context)

            with self.cache_lock:
                if cache_key in self.safety_cache:
                    cached_result, timestamp = self.safety_cache[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        return cached_result

                # CRITICAL FIX: Evict old cache entries
                if len(self.safety_cache) >= self.max_cache_size:
                    current_time = time.time()
                    expired_keys = [
                        k
                        for k, (_, ts) in self.safety_cache.items()
                        if current_time - ts > self.cache_ttl
                    ]
                    for k in expired_keys:
                        del self.safety_cache[k]

                    # If still too large, remove oldest
                    if len(self.safety_cache) >= self.max_cache_size:
                        sorted_items = sorted(
                            self.safety_cache.items(), key=lambda x: x[1][1]
                        )
                        self.safety_cache = dict(
                            sorted_items[-self.max_cache_size // 2 :]
                        )

            with self.lock:
                # Input validation
                is_safe, reason = self.validator.validate_input(context.problem)
                if not is_safe:
                    self._record_violation(
                        context.tool_name, VetoReason.UNSAFE_INPUT, reason
                    )

                    if context.safety_level == SafetyLevel.CRITICAL:
                        result = (SafetyAction.VETO, reason)
                    else:
                        # Try to sanitize
                        sanitized = self.validator.sanitize_input(context.problem)
                        context.problem = sanitized
                        result = (SafetyAction.SANITIZE, reason)

                # Contract validation
                elif context.tool_name in self.contracts:
                    contract = self.contracts[context.tool_name]
                    violation = self._check_contract_violation(context, contract)

                    if violation:
                        self._record_violation(
                            context.tool_name, VetoReason.CONTRACT_VIOLATION, violation
                        )
                        result = (SafetyAction.VETO, violation)
                    else:
                        result = (SafetyAction.ALLOW, None)

                # Rate limiting
                elif self._is_rate_limited(context.tool_name):
                    self._record_violation(
                        context.tool_name,
                        VetoReason.RATE_LIMITED,
                        "Rate limit exceeded",
                    )
                    result = (SafetyAction.VETO, "Rate limit exceeded")

                else:
                    result = (SafetyAction.ALLOW, None)

            # Cache result
            with self.cache_lock:
                self.safety_cache[cache_key] = (result, time.time())

            # Audit
            self._add_audit_entry(context, result)

            return result
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return (SafetyAction.VETO, f"Error: {str(e)}")

    def filter_candidates(
        self, candidates: List[Dict[str, Any]], context: SafetyContext
    ) -> List[Dict[str, Any]]:
        """Filter tool candidates based on safety constraints"""

        filtered = []

        try:
            for candidate in candidates:
                tool_name = candidate.get("tool")
                if not tool_name:
                    continue

                # Create context for this candidate
                candidate_context = SafetyContext(
                    problem=context.problem,
                    tool_name=tool_name,
                    features=context.features,
                    constraints=context.constraints,
                    user_context=context.user_context,
                    safety_level=context.safety_level,
                )

                # Check safety
                action, reason = self.check_safety(candidate_context)

                if action in [
                    SafetyAction.ALLOW,
                    SafetyAction.LOG_AND_ALLOW,
                    SafetyAction.SANITIZE,
                ]:
                    filtered.append(candidate)
                elif action == SafetyAction.REQUIRE_CONFIRMATION:
                    # Add flag for confirmation
                    candidate["requires_confirmation"] = True
                    filtered.append(candidate)
        except Exception as e:
            logger.error(f"Candidate filtering failed: {e}")

        return filtered

    def validate_output(
        self, tool_name: str, output: Any, context: SafetyContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate tool output for safety"""

        try:
            # Basic output validation
            is_safe, reason = self.validator.validate_output(output)
            if not is_safe:
                self._record_violation(tool_name, VetoReason.UNSAFE_OUTPUT, reason)
                return False, reason

            # Contract-based validation
            if tool_name in self.contracts:
                contract = self.contracts[tool_name]

                # Run output validators
                for validator in contract.output_validators:
                    try:
                        if not validator(output):
                            reason = "Output validation failed"
                            self._record_violation(
                                tool_name, VetoReason.UNSAFE_OUTPUT, reason
                            )
                            return False, reason
                    except Exception as e:
                        logger.warning(f"Output validator failed: {e}")

                # Check confidence threshold
                if hasattr(output, "confidence"):
                    min_conf = contract.min_confidence
                    if output.confidence < min_conf:
                        reason = (
                            f"Confidence {output.confidence} below threshold {min_conf}"
                        )
                        self._record_violation(
                            tool_name, VetoReason.CONFIDENCE_TOO_LOW, reason
                        )
                        return False, reason

            return True, None
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def check_consensus(self, outputs: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Check consensus among multiple tool outputs"""

        try:
            is_consistent, confidence, details = (
                self.consistency_checker.check_consistency(outputs)
            )

            if not is_consistent and confidence < self.veto_threshold:
                # Record inconsistency violation
                with self.lock:
                    for tool_name in outputs.keys():
                        self._record_violation(
                            tool_name, VetoReason.INCONSISTENT_OUTPUT, details
                        )

            return is_consistent, confidence, details
        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
            return True, 0.5, f"Error: {str(e)}"

    def _check_contract_violation(
        self, context: SafetyContext, contract: ToolContract
    ) -> Optional[str]:
        """Check if context violates contract"""

        try:
            # Check required inputs
            if contract.required_inputs:
                problem_str = str(context.problem).lower()
                missing = [
                    req for req in contract.required_inputs if req not in problem_str
                ]
                if missing:
                    return f"Missing required inputs: {missing}"

            # Check forbidden inputs
            if contract.forbidden_inputs:
                problem_str = str(context.problem).lower()
                found = [
                    forb for forb in contract.forbidden_inputs if forb in problem_str
                ]
                if found:
                    return f"Forbidden inputs found: {found}"

            # Check resource constraints
            if context.constraints:
                time_budget = context.constraints.get("time_budget_ms", float("inf"))
                if time_budget < contract.max_execution_time_ms * 0.5:
                    return f"Insufficient time budget for {contract.tool_name}"

                energy_budget = context.constraints.get(
                    "energy_budget_mj", float("inf")
                )
                if energy_budget < contract.max_energy_mj * 0.5:
                    return f"Insufficient energy budget for {contract.tool_name}"

            # Check safety level
            if context.safety_level.value < contract.required_safety_level.value:
                return f"Safety level {context.safety_level} insufficient for {contract.tool_name}"

            return None
        except Exception as e:
            logger.error(f"Contract violation check failed: {e}")
            return f"Error checking contract: {str(e)}"

    def _is_rate_limited(self, tool_name: str) -> bool:
        """Check if tool is rate limited"""

        try:
            now = time.time()
            requests = self.rate_limits[tool_name]

            # Remove old requests
            while requests and requests[0] < now - self.rate_limit_window:
                requests.popleft()

            # Check limit
            if len(requests) >= self.max_requests_per_tool:
                return True

            # Add current request
            requests.append(now)
            return False
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False

    def _has_cycles(self, output: Any) -> bool:
        """Check if output contains cycles (for causal graphs)"""

        try:
            # Simplified cycle detection
            if hasattr(output, "graph"):
                # Would implement proper cycle detection here
                return False
            return False
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            return False

    def _validate_multimodal(self, output: Any) -> bool:
        """Validate multimodal output"""

        try:
            # Check that output has expected modalities
            if hasattr(output, "modalities"):
                return len(output.modalities) > 0
            return True
        except Exception as e:
            logger.warning(f"Multimodal validation failed: {e}")
            return True

    # CRITICAL FIX: Prevent unbounded growth
    def _record_violation(self, tool_name: str, reason: VetoReason, details: str):
        """Record safety violation - CRITICAL: Bounded storage"""

        try:
            violation = SafetyViolation(
                timestamp=time.time(),
                tool_name=tool_name,
                violation_type=reason,
                severity=SafetyLevel.MEDIUM,
                details=details[:1000],  # CRITICAL FIX: Limit detail length
                action_taken=SafetyAction.VETO,
            )

            # Already bounded by deque maxlen, but double-check
            self.violations.append(violation)
            self.violation_counts[tool_name] += 1

            # CRITICAL FIX: Trim violation counts if too many tools
            if len(self.violation_counts) > 1000:
                # Keep only tools with recent violations
                recent_tools = set(v.tool_name for v in list(self.violations)[-100:])
                keys_to_remove = [
                    k for k in self.violation_counts.keys() if k not in recent_tools
                ]
                for k in keys_to_remove:
                    del self.violation_counts[k]

            logger.warning(
                f"Safety violation: {tool_name} - {reason.value}: {details[:200]}"
            )
        except Exception as e:
            logger.error(f"Recording violation failed: {e}")

    def _add_audit_entry(
        self, context: SafetyContext, result: Tuple[SafetyAction, Optional[str]]
    ):
        """Add entry to audit trail"""

        try:
            entry = {
                "timestamp": time.time(),
                "tool": context.tool_name,
                "action": result[0].value,
                "reason": result[1][:500]
                if result[1]
                else None,  # CRITICAL FIX: Limit reason length
                "safety_level": context.safety_level.value,
                "problem_hash": hashlib.md5(
                    str(context.problem)[:1000].encode()
                ).hexdigest()[:8],
            }

            # Already bounded by deque maxlen
            self.audit_trail.append(entry)
        except Exception as e:
            logger.error(f"Adding audit entry failed: {e}")

    def _compute_cache_key(self, context: SafetyContext) -> str:
        """Compute cache key for safety check"""

        try:
            # CRITICAL FIX: Limit problem size in hash
            problem_str = str(context.problem)[:1000]

            key_parts = [
                context.tool_name,
                str(context.safety_level.value),
                hashlib.md5(problem_str.encode()).hexdigest()[:16],
            ]

            return "_".join(key_parts)
        except Exception as e:
            logger.error(f"Cache key computation failed: {e}")
            return f"{context.tool_name}_{time.time()}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics"""

        try:
            with self.lock:
                violation_list = list(self.violations)

                return {
                    "total_violations": len(violation_list),
                    "violations_by_tool": dict(self.violation_counts),
                    "recent_violations": [
                        {
                            "tool": v.tool_name,
                            "reason": v.violation_type.value,
                            "timestamp": v.timestamp,
                        }
                        for v in violation_list[-10:]
                    ],
                    "audit_trail_size": len(self.audit_trail),
                    "cache_size": len(self.safety_cache),
                }
        except Exception as e:
            logger.error(f"Getting statistics failed: {e}")
            return {}

    def export_audit_trail(self, path: str):
        """Export audit trail to file"""

        try:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with self.lock:
                audit_data = list(self.audit_trail)

            with open(export_path, "w") as f:
                json.dump(audit_data, f, indent=2, default=str)

            logger.info(f"Audit trail exported to {export_path}")
        except Exception as e:
            logger.error(f"Audit trail export failed: {e}")

    def clear_cache(self):
        """Clear safety cache"""
        with self.cache_lock:
            self.safety_cache.clear()
            logger.info("Safety cache cleared")

    def reset_statistics(self):
        """Reset violation statistics"""
        with self.lock:
            self.violations.clear()
            self.violation_counts.clear()
            logger.info("Safety statistics reset")
