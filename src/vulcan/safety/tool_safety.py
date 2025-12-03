# tool_safety.py
"""
Tool-specific safety management and governance for VULCAN-AGI.
Manages tool safety contracts, usage monitoring, rate limiting, and veto mechanisms.

Revision / Fix Notes (Applied):
1. Removed merge conflict markers (<<<<<<< HEAD / ======= / >>>>>>>).
2. Preserved all original logic; only conflict artifacts removed.
3. Returned full, untruncated file per request.
4. FIXED: Made atexit handlers non-blocking during pytest runs to prevent test freeze.
"""

import logging
import threading
import time
import json
import atexit
import os
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .safety_types import (
    ToolSafetyLevel,
    ToolSafetyContract,
    SafetyViolationType,
    SafetyReport,
    ActionType,
    Condition
)

logger = logging.getLogger(__name__)

_TOOL_SAFETY_INIT_DONE = False

# Helper function for safe logging during shutdown
def safe_log(log_func, message):
    """Log safely during shutdown when logging may be closed."""
    try:
        log_func(message)
    except (ValueError, AttributeError, OSError, RuntimeError):
        pass

# ============================================================
# TOKEN BUCKET RATE LIMITER
# ============================================================

class TokenBucket:
    """Token bucket for efficient rate limiting."""
    
    def __init__(self, rate: float, capacity: float):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second
            capacity: Maximum token capacity
        """
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.RLock()
        self._shutdown = False
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens. Returns True if successful.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        if self._shutdown:
            return False
        
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_available(self) -> float:
        """
        Get number of available tokens.
        
        Returns:
            Number of tokens currently available
        """
        if self._shutdown:
            return 0.0
        
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.capacity, self.tokens + elapsed * self.rate)
    
    def shutdown(self):
        """Shutdown token bucket."""
        logging.raiseExceptions = False
        self._shutdown = True

# ============================================================
# TOOL SAFETY MANAGER
# ============================================================

class ToolSafetyManager:
    """
    Manages tool-specific safety contracts and vetos.
    Provides rate limiting, resource tracking, and safety scoring for tools.
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tool safety manager with configuration."""
        # Reset shutdown flag even if already initialized (for tests)
        self._shutdown = False
        
        if getattr(self, "_initialized", False):
            return
        
        self.config = config or {}
        self.contracts = {}
        self.usage_history = defaultdict(lambda: deque(maxlen=1000))
        self.veto_history = deque(maxlen=1000)
        self.contract_violations = defaultdict(int)
        self.safety_scores = defaultdict(float)
        self.rate_limiters = {}
        self.resource_usage_tracker = defaultdict(lambda: deque(maxlen=100))
        self.performance_metrics = defaultdict(dict)
        self.lock = threading.RLock()
        
        self.max_veto_history = self.config.get('max_veto_history', 1000)
        self.safety_score_decay = self.config.get('safety_score_decay', 0.95)
        self.violation_penalty = self.config.get('violation_penalty', 0.1)
        
        self._initialize_default_contracts()
        self._initialize_rate_limiters()
        atexit.register(self.shutdown)
        
        logger.info("ToolSafetyManager initialized")
        self._initialized = True
        
    def _initialize_rate_limiters(self):
        """Initialize token bucket rate limiters for all contracts."""
        with self.lock:
            for tool_name, contract in self.contracts.items():
                rate_per_second = contract.max_frequency / 60.0
                capacity = contract.max_frequency
                self.rate_limiters[tool_name] = TokenBucket(rate_per_second, capacity)
        
    def _initialize_default_contracts(self):
        """Initialize default safety contracts for common tools."""
        
        self.contracts['probabilistic'] = ToolSafetyContract(
            tool_name='probabilistic',
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[
                Condition('confidence', '>', 0.3, "Minimum confidence required"),
                Condition('data_quality', '>', 0.5, "Minimum data quality"),
                Condition('corrupted_data', '==', False, "Data must not be corrupted")
            ],
            postconditions=[
                Condition('uncertainty', '<', 0.9, "Uncertainty must be acceptable"),
                Condition('valid_distribution', '==', True, "Must produce valid distribution")
            ],
            invariants=[
                lambda: self._check_system_stability(),
                lambda: self._check_memory_availability()
            ],
            max_frequency=100.0,
            max_resource_usage={'memory_mb': 1000, 'time_ms': 5000, 'cpu_percent': 50},
            required_confidence=0.5,
            veto_conditions=[
                Condition('adversarial_detected', '==', True, "No adversarial inputs"),
                Condition('system_overload', '==', True, "System not overloaded")
            ],
            risk_score=0.2,
            description="Probabilistic reasoning with uncertainty quantification",
            metadata={'category': 'reasoning', 'version': '2.0'}
        )
        
        self.contracts['symbolic'] = ToolSafetyContract(
            tool_name='symbolic',
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[
                Condition('logic_valid', '==', True, "Logic must be valid"),
                Condition('axioms_count', '>', 0, "Must have axioms"),
                Condition('axioms_count', '<', 1000, "Prevent axiom explosion"),
                Condition('contradictory_axioms', '==', False, "No contradictions")
            ],
            postconditions=[
                Condition('proof_valid', '==', True, "Proof must be valid or low confidence"),
                Condition('infinite_recursion', '==', False, "No infinite recursion"),
                Condition('steps', '<', 10000, "Step limit")
            ],
            invariants=[
                lambda: self._check_logic_consistency(),
                lambda: not self._detect_circular_reasoning()
            ],
            max_frequency=50.0,
            max_resource_usage={'memory_mb': 2000, 'time_ms': 10000, 'cpu_percent': 70},
            required_confidence=0.6,
            veto_conditions=[
                Condition('infinite_loop_risk', '==', True, "No infinite loop risk"),
                Condition('halting_problem_detected', '==', True, "No halting problem")
            ],
            risk_score=0.3,
            description="Symbolic logic reasoning with formal proofs",
            metadata={'category': 'reasoning', 'complexity': 'high'}
        )
        
        self.contracts['causal'] = ToolSafetyContract(
            tool_name='causal',
            safety_level=ToolSafetyLevel.SUPERVISED,
            preconditions=[
                Condition('causal_graph_valid', '==', True, "Causal graph must be valid"),
                Condition('sample_size', '>', 30, "Minimum sample size"),
                Condition('temporal_paradox', '==', False, "No temporal paradox")
            ],
            postconditions=[
                Condition('causal_strength', '>', 0.1, "Meaningful causal strength"),
                Condition('p_value', '<', 0.05, "Statistical significance"),
                Condition('spurious_correlation', '==', False, "No spurious correlation")
            ],
            invariants=[
                lambda: self._check_causality_assumptions(),
                lambda: not self._detect_confounding_variables()
            ],
            max_frequency=30.0,
            max_resource_usage={'memory_mb': 3000, 'time_ms': 15000, 'cpu_percent': 60},
            required_confidence=0.7,
            veto_conditions=[
                Condition('spurious_correlation_risk', '==', True, "No spurious correlation risk"),
                Condition('simpson_paradox_detected', '==', True, "No Simpson's paradox")
            ],
            risk_score=0.4,
            description="Causal inference and relationship analysis",
            metadata={'category': 'reasoning', 'statistical': True}
        )
        
        self.contracts['portfolio'] = ToolSafetyContract(
            tool_name='portfolio',
            safety_level=ToolSafetyLevel.RESTRICTED,
            preconditions=[
                Condition('num_tools', '<=', 5, "Maximum 5 tools"),
                Condition('total_resource_budget', '<', 10000, "Resource budget limit"),
                Condition('conflicting_tools', '==', False, "No conflicting tools")
            ],
            postconditions=[
                Condition('execution_time', '<', 30000, "Maximum execution time"),
                Condition('resource_overflow', '==', False, "No resource overflow")
            ],
            invariants=[
                lambda: self._check_portfolio_invariant(),
                lambda: self._check_no_deadlocks()
            ],
            max_frequency=10.0,
            max_resource_usage={'memory_mb': 5000, 'time_ms': 30000, 'cpu_percent': 80},
            required_confidence=0.8,
            veto_conditions=[
                Condition('emergency_stop', '==', True, "No emergency stop"),
                Condition('resource_exhaustion', '==', True, "No resource exhaustion")
            ],
            risk_score=0.6,
            description="Multi-tool portfolio execution and orchestration",
            metadata={'category': 'execution', 'parallel': True}
        )
        
        self.contracts['neural'] = ToolSafetyContract(
            tool_name='neural',
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[
                Condition('model_loaded', '==', True, "Model must be loaded"),
                Condition('input_shape_valid', '==', True, "Valid input shape"),
                Condition('adversarial_input', '==', False, "No adversarial input")
            ],
            postconditions=[
                Condition('output_valid', '==', True, "Valid output"),
                Condition('confidence', '>', 0.1, "Minimum confidence")
            ],
            invariants=[
                lambda: self._check_gpu_memory(),
                lambda: self._check_model_stability()
            ],
            max_frequency=200.0,
            max_resource_usage={'memory_mb': 4000, 'time_ms': 2000, 'gpu_memory_mb': 2000},
            required_confidence=0.4,
            veto_conditions=[
                Condition('gpu_unavailable', '==', True, "GPU must be available"),
                Condition('model_corrupted', '==', True, "Model must not be corrupted")
            ],
            risk_score=0.25,
            description="Neural network inference and prediction",
            metadata={'category': 'ml', 'requires_gpu': True}
        )
        
        self.contracts['data_processor'] = ToolSafetyContract(
            tool_name='data_processor',
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[
                Condition('data_size_mb', '<', 1000, "Maximum data size"),
                Condition('data_format_valid', '==', True, "Valid data format")
            ],
            postconditions=[
                Condition('processing_complete', '==', True, "Processing complete"),
                Condition('data_integrity_maintained', '==', True, "Data integrity maintained")
            ],
            invariants=[
                lambda: self._check_disk_space(),
                lambda: not self._detect_data_corruption()
            ],
            max_frequency=50.0,
            max_resource_usage={'memory_mb': 2000, 'time_ms': 10000, 'disk_mb': 5000},
            required_confidence=0.5,
            veto_conditions=[
                Condition('malicious_data_detected', '==', True, "No malicious data"),
                Condition('privacy_violation', '==', True, "No privacy violation")
            ],
            risk_score=0.3,
            description="Data processing and transformation pipeline",
            metadata={'category': 'data', 'batch_capable': True}
        )
    
    def register_contract(self, contract: ToolSafetyContract):
        """Register a new tool safety contract."""
        if self._shutdown:
            logger.warning("Cannot register contract - manager is shut down")
            return
        
        with self.lock:
            self.contracts[contract.tool_name] = contract
            self.safety_scores[contract.tool_name] = 1.0 - contract.risk_score
            rate_per_second = contract.max_frequency / 60.0
            capacity = contract.max_frequency
            self.rate_limiters[contract.tool_name] = TokenBucket(rate_per_second, capacity)
            logger.info(f"Registered safety contract for tool: {contract.tool_name} "
                       f"with safety level: {contract.safety_level.value}")
    
    def unregister_contract(self, tool_name: str) -> bool:
        """Unregister a tool safety contract."""
        if self._shutdown:
            logger.warning("Cannot unregister contract - manager is shut down")
            return False
        
        with self.lock:
            if tool_name in self.contracts:
                del self.contracts[tool_name]
                if tool_name in self.safety_scores:
                    del self.safety_scores[tool_name]
                if tool_name in self.rate_limiters:
                    self.rate_limiters[tool_name].shutdown()
                    del self.rate_limiters[tool_name]
                logger.info(f"Unregistered safety contract for tool: {tool_name}")
                return True
            return False
    
    def check_tool_safety(self, tool_name: str, context: Dict[str, Any]) -> Tuple[bool, SafetyReport]:
        """
        Check if tool usage is safe according to its contract.
        """
        if self._shutdown:
            return False, SafetyReport(
                safe=False,
                confidence=0.0,
                violations=[SafetyViolationType.VALIDATION_ERROR],
                reasons=["Tool safety manager is shut down"]
            )
        
        if tool_name not in self.contracts:
            logger.warning(f"No safety contract for tool: {tool_name} - using default safety")
            return True, SafetyReport(
                safe=True,
                confidence=0.5,
                metadata={'tool': tool_name, 'warning': 'No safety contract defined'}
            )
        
        contract = self.contracts[tool_name]
        violations = []
        reasons = []
        vetoed = False
        
        if contract.safety_level == ToolSafetyLevel.PROHIBITED:
            violations.append(SafetyViolationType.TOOL_CONTRACT)
            reasons.append(f"Tool {tool_name} is prohibited")
            vetoed = True
        
        if not vetoed:
            try:
                precond_valid, precond_failures = contract.validate_preconditions(context)
                if not precond_valid:
                    violations.append(SafetyViolationType.TOOL_CONTRACT)
                    reasons.extend(precond_failures)
            except Exception as e:
                logger.error(f"Error checking preconditions for {tool_name}: {e}")
                violations.append(SafetyViolationType.TOOL_CONTRACT)
                reasons.append(f"Precondition check failed: {str(e)}")
        
        try:
            veto_triggered, veto_reasons = contract.check_veto(context)
            if veto_triggered:
                violations.append(SafetyViolationType.TOOL_VETO)
                reasons.extend(veto_reasons)
                vetoed = True
        except Exception as e:
            logger.error(f"Error checking veto conditions for {tool_name}: {e}")
        
        if not self._check_rate_limit(tool_name):
            violations.append(SafetyViolationType.TOOL_CONTRACT)
            reasons.append(f"Rate limit exceeded for {tool_name} "
                          f"(max {contract.max_frequency} calls/min)")
        
        resource_usage = context.get('estimated_resources', {})
        for resource, limit in contract.max_resource_usage.items():
            if resource in resource_usage and resource_usage[resource] > limit:
                violations.append(SafetyViolationType.TOOL_CONTRACT)
                reasons.append(f"Resource limit exceeded for {tool_name}: "
                              f"{resource} ({resource_usage[resource]} > {limit})")
        
        confidence = context.get('confidence', 0)
        if confidence < contract.required_confidence:
            violations.append(SafetyViolationType.TOOL_CONTRACT)
            reasons.append(f"Insufficient confidence for {tool_name}: "
                          f"{confidence:.2f} < {contract.required_confidence}")
        
        for i, invariant in enumerate(contract.invariants):
            try:
                if not invariant():
                    violations.append(SafetyViolationType.TOOL_CONTRACT)
                    reasons.append(f"Invariant {i} violated for {tool_name}")
            except Exception as e:
                logger.error(f"Error checking invariant {i} for {tool_name}: {e}")
                violations.append(SafetyViolationType.TOOL_CONTRACT)
                reasons.append(f"Invariant {i} check failed for {tool_name}")
        
        safe = len(violations) == 0
        
        with self.lock:
            self._update_usage_history(tool_name, safe, context, violations)
            if not safe:
                self.contract_violations[tool_name] += 1
            if vetoed:
                self._record_veto(tool_name, reasons)
        
        safety_score = self._calculate_safety_score(tool_name, contract, violations)
        with self.lock:
            self.safety_scores[tool_name] = safety_score
        
        self._update_performance_metrics(tool_name, safe, len(violations))
        
        report = SafetyReport(
            safe=safe,
            confidence=1.0 - contract.risk_score if safe else contract.risk_score,
            violations=violations,
            reasons=reasons,
            tool_vetoes=[tool_name] if vetoed else [],
            metadata={
                'tool_name': tool_name,
                'safety_level': contract.safety_level.value,
                'safety_score': safety_score,
                'vetoed': vetoed,
                'rate_limit_remaining': self._get_rate_limit_remaining(tool_name),
                'resource_usage': resource_usage
            }
        )
        
        return safe, report
    
    def check_postconditions(self, tool_name: str, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check postconditions after tool execution."""
        if self._shutdown:
            return False, ["Tool safety manager is shut down"]
        
        if tool_name not in self.contracts:
            return True, []
        
        contract = self.contracts[tool_name]
        
        try:
            postcond_valid, failures = contract.validate_postconditions(result)
            if not postcond_valid:
                with self.lock:
                    self.contract_violations[tool_name] += len(failures)
                    self.safety_scores[tool_name] *= (1.0 - self.violation_penalty)
            
            if 'actual_resources' in result:
                self._record_resource_usage(tool_name, result['actual_resources'])
            
            return postcond_valid, failures
        except Exception as e:
            logger.error(f"Error checking postconditions for {tool_name}: {e}")
            return False, [f"Postcondition check failed: {str(e)}"]
    
    def veto_tool_selection(self, tool_names: List[str], 
                           context: Dict[str, Any]) -> Tuple[List[str], SafetyReport]:
        """Apply safety veto to a list of tools."""
        if self._shutdown:
            return [], SafetyReport(
                safe=False,
                confidence=0.0,
                violations=[SafetyViolationType.VALIDATION_ERROR],
                reasons=["Tool safety manager is shut down"]
            )
        
        allowed_tools = []
        all_violations = []
        all_reasons = []
        vetoed_tools = []
        individual_reports = []
        
        for tool_name in tool_names:
            safe, report = self.check_tool_safety(tool_name, context)
            individual_reports.append(report)
            
            if safe:
                allowed_tools.append(tool_name)
            else:
                vetoed_tools.append(tool_name)
                all_violations.extend(report.violations)
                all_reasons.extend(report.reasons)
                logger.warning(f"Tool {tool_name} vetoed: {report.reasons}")
        
        if len(allowed_tools) > 1:
            compatible_tools = self._filter_compatible_tools(allowed_tools, context)
            incompatible = set(allowed_tools) - set(compatible_tools)
            if incompatible:
                vetoed_tools.extend(list(incompatible))
                all_reasons.append(f"Tools {incompatible} incompatible with others")
            allowed_tools = compatible_tools
        
        combined_report = SafetyReport(
            safe=len(vetoed_tools) == 0,
            confidence=0.9 if len(vetoed_tools) == 0 else 0.4,
            violations=all_violations,
            reasons=all_reasons,
            tool_vetoes=vetoed_tools,
            metadata={
                'total_tools': len(tool_names),
                'allowed_tools': len(allowed_tools),
                'vetoed_tools': len(vetoed_tools),
                'individual_reports': [r.to_audit_log() for r in individual_reports]
            }
        )
        
        return allowed_tools, combined_report
    
    def _check_rate_limit(self, tool_name: str, tokens: float = 1.0) -> bool:
        with self.lock:
            if tool_name not in self.rate_limiters:
                if tool_name in self.contracts:
                    contract = self.contracts[tool_name]
                    rate_per_second = contract.max_frequency / 60.0
                    self.rate_limiters[tool_name] = TokenBucket(
                        rate_per_second,
                        contract.max_frequency
                    )
                else:
                    return True
            return self.rate_limiters[tool_name].consume(tokens)
    
    def _get_rate_limit_remaining(self, tool_name: str) -> int:
        with self.lock:
            if tool_name not in self.rate_limiters:
                return int(self.contracts.get(tool_name, ToolSafetyContract(
                    tool_name='default',
                    safety_level=ToolSafetyLevel.MONITORED,
                    max_frequency=100.0
                )).max_frequency)
            return int(self.rate_limiters[tool_name].get_available())
    
    def _check_portfolio_invariant(self) -> bool:
        with self.lock:
            active_high_risk = sum(
                1 for tool, contract in self.contracts.items()
                if contract.risk_score > 0.5 and self._is_tool_active(tool)
            )
        return active_high_risk <= 3
    
    def _is_tool_active(self, tool_name: str) -> bool:
        if tool_name not in self.usage_history:
            return False
        recent = [h for h in self.usage_history[tool_name] 
                 if h['timestamp'] > time.time() - 60]
        return len(recent) > 0
    
    def _calculate_safety_score(self, tool_name: str, contract: ToolSafetyContract, 
                               violations: List[SafetyViolationType]) -> float:
        base_score = 1.0 - contract.risk_score
        violation_penalty = len(violations) * self.violation_penalty
        
        with self.lock:
            if tool_name in self.contract_violations:
                total_uses = len(self.usage_history.get(tool_name, []))
                if total_uses > 0:
                    violation_rate = self.contract_violations[tool_name] / total_uses
                    history_penalty = min(0.3, violation_rate * 0.5)
                else:
                    history_penalty = 0
            else:
                history_penalty = 0
            
            if tool_name in self.safety_scores:
                previous_score = self.safety_scores[tool_name]
                decayed_score = previous_score * self.safety_score_decay
            else:
                decayed_score = base_score
        
        new_score = max(0.0, min(1.0, 
            decayed_score - violation_penalty - history_penalty))
        return new_score
    
    def _update_usage_history(self, tool_name: str, safe: bool, 
                             context: Dict[str, Any], violations: List):
        self.usage_history[tool_name].append({
            'timestamp': time.time(),
            'safe': safe,
            'context': context,
            'violations': [v.value for v in violations]
        })
    
    def _record_veto(self, tool_name: str, reasons: List[str]):
        self.veto_history.append({
            'tool': tool_name,
            'timestamp': time.time(),
            'reasons': reasons
        })
    
    def _record_resource_usage(self, tool_name: str, resources: Dict[str, float]):
        with self.lock:
            self.resource_usage_tracker[tool_name].append({
                'timestamp': time.time(),
                'resources': resources
            })
    
    def _update_performance_metrics(self, tool_name: str, safe: bool, violation_count: int):
        with self.lock:
            if tool_name not in self.performance_metrics:
                self.performance_metrics[tool_name] = {
                    'total_uses': 0,
                    'safe_uses': 0,
                    'total_violations': 0,
                    'last_updated': time.time()
                }
            metrics = self.performance_metrics[tool_name]
            metrics['total_uses'] += 1
            if safe:
                metrics['safe_uses'] += 1
            metrics['total_violations'] += violation_count
            metrics['last_updated'] = time.time()
    
    def _check_tool_compatibility(self, tools: List[str]) -> bool:
        incompatible_pairs = [
            ('fast_heuristic', 'deep_analysis'),
            ('greedy_search', 'exhaustive_search'),
            ('synchronous', 'asynchronous')
        ]
        for tool1, tool2 in incompatible_pairs:
            if tool1 in tools and tool2 in tools:
                return False
        return True
    
    def _filter_compatible_tools(self, tools: List[str], context: Dict[str, Any]) -> List[str]:
        compatible = tools.copy()
        incompatible_pairs = [
            ('fast_heuristic', 'deep_analysis'),
            ('greedy_search', 'exhaustive_search'),
            ('synchronous', 'asynchronous')
        ]
        with self.lock:
            for tool1, tool2 in incompatible_pairs:
                if tool1 in compatible and tool2 in compatible:
                    score1 = self.safety_scores.get(tool1, 0)
                    score2 = self.safety_scores.get(tool2, 0)
                    if score1 > score2:
                        compatible.remove(tool2)
                    else:
                        compatible.remove(tool1)
        return compatible
    
    def _check_system_stability(self) -> bool:
        if self._shutdown:
            return False
        with self.lock:
            recent_uses = sum(
                len([h for h in history if h['timestamp'] > time.time() - 300])
                for history in self.usage_history.values()
            )
            recent_failures = sum(
                len([h for h in history 
                     if h['timestamp'] > time.time() - 300 and not h['safe']])
                for history in self.usage_history.values()
            )
            if recent_uses == 0:
                return True
            return (recent_failures / recent_uses) < 0.5
    
    def _check_memory_availability(self) -> bool:
        return not self._shutdown
    
    def _check_logic_consistency(self) -> bool:
        return not self._shutdown
    
    def _detect_circular_reasoning(self) -> bool:
        return False
    
    def _check_causality_assumptions(self) -> bool:
        return not self._shutdown
    
    def _detect_confounding_variables(self) -> bool:
        return False
    
    def _check_no_deadlocks(self) -> bool:
        return not self._shutdown
    
    def _check_gpu_memory(self) -> bool:
        return not self._shutdown
    
    def _check_model_stability(self) -> bool:
        return not self._shutdown
    
    def _check_disk_space(self) -> bool:
        return not self._shutdown
    
    def _detect_data_corruption(self) -> bool:
        return False
    
    def get_tool_safety_report(self, tool_name: str) -> Dict[str, Any]:
        if self._shutdown:
            return {'status': 'shutdown', 'tool': tool_name}
        
        with self.lock:
            if tool_name not in self.contracts:
                return {'status': 'no_contract', 'tool': tool_name}
            
            contract = self.contracts[tool_name]
            usage_count = len(self.usage_history[tool_name])
            violation_count = self.contract_violations[tool_name]
            violation_rate = violation_count / max(1, usage_count)
            recent_vetos = [v for v in self.veto_history 
                           if v['tool'] == tool_name and 
                           v['timestamp'] > time.time() - 3600]
            recent_resources = self.resource_usage_tracker[tool_name]
            if recent_resources:
                avg_resources = {}
                for key in recent_resources[0].get('resources', {}).keys():
                    values = [r['resources'].get(key, 0) for r in recent_resources]
                    avg_resources[key] = sum(values) / len(values)
            else:
                avg_resources = {}
            perf_metrics = self.performance_metrics.get(tool_name, {})
            contract_details = {
                'tool_name': contract.tool_name,
                'safety_level': contract.safety_level.value,
                'description': contract.description,
                'risk_score': contract.risk_score,
                'max_frequency': contract.max_frequency,
                'max_resource_usage': contract.max_resource_usage,
                'required_confidence': contract.required_confidence,
                'preconditions_count': len(contract.preconditions),
                'postconditions_count': len(contract.postconditions),
                'invariants_count': len(contract.invariants),
                'veto_conditions_count': len(contract.veto_conditions),
                'metadata': contract.metadata
            }
            
            return {
                'tool_name': tool_name,
                'safety_level': contract.safety_level.value,
                'risk_score': contract.risk_score,
                'current_safety_score': self.safety_scores.get(tool_name, 0),
                'usage_count': usage_count,
                'violation_count': violation_count,
                'violation_rate': violation_rate,
                'recent_vetos': len(recent_vetos),
                'max_frequency': contract.max_frequency,
                'required_confidence': contract.required_confidence,
                'rate_limit_remaining': self._get_rate_limit_remaining(tool_name),
                'average_resource_usage': avg_resources,
                'performance': {
                    'safety_rate': perf_metrics.get('safe_uses', 0) / max(1, perf_metrics.get('total_uses', 1)),
                    'total_uses': perf_metrics.get('total_uses', 0)
                },
                'contract_details': contract_details,
                'metadata': contract.metadata
            }
    
    def update_contract_risk(self, tool_name: str, new_risk_score: Optional[float] = None):
        if self._shutdown:
            return
        
        if tool_name not in self.contracts:
            logger.warning(f"Cannot update risk for unknown tool: {tool_name}")
            return
        
        contract = self.contracts[tool_name]
        
        with self.lock:
            if new_risk_score is not None:
                old_score = contract.risk_score
                contract.risk_score = max(0.0, min(1.0, new_risk_score))
                logger.info(f"Updated risk score for {tool_name}: {old_score:.2f} -> {contract.risk_score:.2f}")
            else:
                usage_count = len(self.usage_history[tool_name])
                if usage_count > 10:
                    violation_rate = self.contract_violations[tool_name] / usage_count
                    old_score = contract.risk_score
                    if violation_rate > 0.2:
                        contract.risk_score = min(1.0, contract.risk_score * 1.1)
                    elif violation_rate < 0.05:
                        contract.risk_score = max(0.1, contract.risk_score * 0.95)
                    if old_score != contract.risk_score:
                        logger.info(f"Auto-adjusted risk score for {tool_name}: "
                                   f"{old_score:.2f} -> {contract.risk_score:.2f} "
                                   f"(violation rate: {violation_rate:.2%})")
            
            self.safety_scores[tool_name] = 1.0 - contract.risk_score
    
    def get_global_safety_stats(self) -> Dict[str, Any]:
        if self._shutdown:
            return {'status': 'shutdown'}
        
        with self.lock:
            total_uses = sum(len(h) for h in self.usage_history.values())
            total_violations = sum(self.contract_violations.values())
            total_vetos = len(self.veto_history)
            tool_stats = []
            for tool_name in self.contracts:
                stats = self.get_tool_safety_report(tool_name)
                if 'status' not in stats:
                    tool_stats.append({
                        'tool': tool_name,
                        'safety_score': stats['current_safety_score'],
                        'violation_rate': stats['violation_rate'],
                        'usage_count': stats['usage_count']
                    })
            tool_stats.sort(key=lambda x: x['usage_count'], reverse=True)
            
            return {
                'total_tool_uses': total_uses,
                'total_violations': total_violations,
                'total_vetos': total_vetos,
                'global_violation_rate': total_violations / max(1, total_uses),
                'active_contracts': len(self.contracts),
                'average_safety_score': sum(self.safety_scores.values()) / max(1, len(self.safety_scores)),
                'most_used_tools': tool_stats[:5],
                'highest_risk_tools': sorted(
                    [(t, c.risk_score) for t, c in self.contracts.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]
            }
    
    def shutdown(self):
        if self._shutdown:
            return
        
        # FIXED: Skip blocking operations during pytest runs
        is_pytest = os.environ.get("PYTEST_RUNNING") == "1"
        if is_pytest:
            self._shutdown = True
            return
        
        logging.raiseExceptions = False
        safe_log(logger.info, "Shutting down ToolSafetyManager...")
        self._shutdown = True
        with self.lock:
            for limiter in self.rate_limiters.values():
                limiter.shutdown()
            self.contracts.clear()
            self.usage_history.clear()
            self.veto_history.clear()
            self.contract_violations.clear()
            self.safety_scores.clear()
            self.rate_limiters.clear()
            self.resource_usage_tracker.clear()
            self.performance_metrics.clear()
        safe_log(logger.info, "ToolSafetyManager shutdown complete")

# ============================================================
# TOOL SAFETY GOVERNOR
# ============================================================

class ToolSafetyGovernor:
    """
    High-level safety governance for tool selection and execution.
    Provides emergency controls, consensus requirements, and governance policies.
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        
        self.config = config or {}
        self.tool_safety_manager = ToolSafetyManager(config)
        self.veto_threshold = self.config.get('veto_threshold', 0.8)
        self.require_consensus = self.config.get('require_consensus', False)
        self.governance_history = deque(maxlen=1000)
        self.emergency_stop = False
        self.emergency_stop_reason = None
        self.quarantine_list = set()
        self.whitelist = set(self.config.get('whitelist', []))
        self.blacklist = set(self.config.get('blacklist', []))
        self.governance_policies = self._initialize_policies()
        self.lock = threading.RLock()
        self._shutdown = False
        
        self.quarantine_threads = []
        self.governance_metrics = {
            'total_requests': 0,
            'approved_requests': 0,
            'vetoed_requests': 0,
            'emergency_stops': 0,
            'consensus_failures': 0
        }
        atexit.register(self.shutdown)
        logger.info("ToolSafetyGovernor initialized")
        self._initialized = True
    
    def _initialize_policies(self) -> Dict[str, Any]:
        return {
            'max_parallel_tools': self.config.get('max_parallel_tools', 5),
            'require_approval_above_risk': self.config.get('require_approval_above_risk', 0.7),
            'auto_quarantine_on_failures': self.config.get('auto_quarantine_on_failures', 3),
            'consensus_threshold': self.config.get('consensus_threshold', 0.6),
            'allow_experimental': self.config.get('allow_experimental', False)
        }
    
    def govern_tool_selection(self, selection_request: Dict[str, Any], 
                            selected_tools: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        if self._shutdown:
            return [], {'status': 'shutdown', 'allowed_tools': []}
        
        with self.lock:
            self.governance_metrics['total_requests'] += 1
        
        if self.emergency_stop:
            logger.critical(f"Emergency stop active: {self.emergency_stop_reason}")
            with self.lock:
                self.governance_metrics['vetoed_requests'] += 1
            return [], {
                'status': 'emergency_stop',
                'reason': self.emergency_stop_reason,
                'allowed_tools': []
            }
        
        filtered_tools = self._apply_lists(selected_tools)
        with self.lock:
            filtered_tools = [t for t in filtered_tools if t not in self.quarantine_list]
        
        if len(filtered_tools) > self.governance_policies['max_parallel_tools']:
            filtered_tools = self._select_best_tools(
                filtered_tools, 
                self.governance_policies['max_parallel_tools']
            )
        
        context = {
            'confidence': selection_request.get('confidence', 0),
            'constraints': selection_request.get('constraints', {}),
            'features': selection_request.get('features'),
            'estimated_resources': self._estimate_resources(filtered_tools),
            'num_tools': len(filtered_tools),
            'tools': filtered_tools
        }
        
        allowed_tools, veto_report = self.tool_safety_manager.veto_tool_selection(
            filtered_tools, context
        )
        
        high_risk_tools = self._identify_high_risk_tools(allowed_tools)
        if high_risk_tools and not selection_request.get('risk_approved', False):
            logger.warning(f"High-risk tools require approval: {high_risk_tools}")
            allowed_tools = [t for t in allowed_tools if t not in high_risk_tools]
            veto_report.reasons.append(f"High-risk tools removed: {high_risk_tools}")
        
        consensus_tools = allowed_tools
        if self.require_consensus and len(allowed_tools) > 1:
            consensus_tools = self._ensure_consensus(allowed_tools, context)
            if len(consensus_tools) < len(allowed_tools):
                with self.lock:
                    self.governance_metrics['consensus_failures'] += 1
                veto_report.reasons.append("Some tools removed for consensus")
            allowed_tools = consensus_tools
        
        with self.lock:
            if allowed_tools:
                self.governance_metrics['approved_requests'] += 1
            else:
                self.governance_metrics['vetoed_requests'] += 1
        
        governance_record = {
            'timestamp': time.time(),
            'requested_tools': selected_tools,
            'allowed_tools': allowed_tools,
            'vetoed_tools': veto_report.tool_vetoes,
            'reasons': veto_report.reasons,
            'policies_applied': list(self.governance_policies.keys())
        }
        
        with self.lock:
            self.governance_history.append(governance_record)
        
        return allowed_tools, {
            'allowed_tools': allowed_tools,
            'veto_report': veto_report.to_audit_log(),
            'governance_record': governance_record,
            'high_risk_tools': high_risk_tools,
            'consensus_achieved': len(consensus_tools) == len(allowed_tools) if self.require_consensus else True
        }
    
    def validate_execution_result(self, tool_name: str, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if self._shutdown:
            return False, ["Tool safety governor is shut down"]
        
        valid, failures = self.tool_safety_manager.check_postconditions(tool_name, result)
        if not valid:
            self._track_tool_failure(tool_name)
        return valid, failures
    
    def trigger_emergency_stop(self, reason: str):
        if self._shutdown:
            return
        with self.lock:
            self.emergency_stop = True
            self.emergency_stop_reason = reason
            self.governance_metrics['emergency_stops'] += 1
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            self.governance_history.append({
                'timestamp': time.time(),
                'event': 'emergency_stop',
                'reason': reason
            })
    
    def clear_emergency_stop(self, authorized_by: str):
        if self._shutdown:
            return
        with self.lock:
            if not self.emergency_stop:
                logger.warning("No emergency stop to clear")
                return
            self.emergency_stop = False
            previous_reason = self.emergency_stop_reason
            self.emergency_stop_reason = None
            logger.info(f"Emergency stop cleared by: {authorized_by} (was: {previous_reason})")
            self.governance_history.append({
                'timestamp': time.time(),
                'event': 'emergency_stop_cleared',
                'authorized_by': authorized_by,
                'previous_reason': previous_reason
            })
    
    def quarantine_tool(self, tool_name: str, reason: str, duration_seconds: float = 3600):
        if self._shutdown:
            return
        with self.lock:
            self.quarantine_list.add(tool_name)
            logger.warning(f"Tool {tool_name} quarantined for {duration_seconds}s: {reason}")
            self.governance_history.append({
                'timestamp': time.time(),
                'event': 'tool_quarantine',
                'tool': tool_name,
                'reason': reason,
                'duration': duration_seconds
            })
        
        def remove_quarantine():
            time.sleep(duration_seconds)
            if not self._shutdown:
                with self.lock:
                    self.quarantine_list.discard(tool_name)
                logger.info(f"Tool {tool_name} removed from quarantine")
        
        thread = threading.Thread(target=remove_quarantine, daemon=True, name=f"Quarantine-{tool_name}")
        thread.start()
        with self.lock:
            self.quarantine_threads.append(thread)
            self.quarantine_threads = [t for t in self.quarantine_threads if t.is_alive()]
    
    def _apply_lists(self, tools: List[str]) -> List[str]:
        filtered = tools.copy()
        with self.lock:
            if self.blacklist:
                filtered = [t for t in filtered if t not in self.blacklist]
            if self.whitelist:
                filtered = [t for t in filtered if t in self.whitelist]
        return filtered
    
    def _estimate_resources(self, tools: List[str]) -> Dict[str, float]:
        total_resources = {
            'memory_mb': 0,
            'time_ms': 0,
            'cpu_percent': 0,
            'gpu_memory_mb': 0,
            'disk_mb': 0
        }
        for tool in tools:
            if tool in self.tool_safety_manager.contracts:
                contract = self.tool_safety_manager.contracts[tool]
                for resource, usage in contract.max_resource_usage.items():
                    if resource in total_resources:
                        total_resources[resource] += usage
        return total_resources
    
    def _ensure_consensus(self, tools: List[str], context: Dict[str, Any]) -> List[str]:
        conflicting_pairs = [
            ('fast_heuristic', 'deep_analysis'),
            ('greedy_search', 'exhaustive_search'),
            ('synchronous', 'asynchronous'),
            ('probabilistic', 'deterministic')
        ]
        filtered_tools = tools.copy()
        for tool1, tool2 in conflicting_pairs:
            if tool1 in filtered_tools and tool2 in filtered_tools:
                score1 = self.tool_safety_manager.safety_scores.get(tool1, 0)
                score2 = self.tool_safety_manager.safety_scores.get(tool2, 0)
                if score1 > score2:
                    filtered_tools.remove(tool2)
                else:
                    filtered_tools.remove(tool1)
        if len(filtered_tools) > 1:
            avg_safety = sum(
                self.tool_safety_manager.safety_scores.get(t, 0.5) 
                for t in filtered_tools
            ) / len(filtered_tools)
            if avg_safety < self.governance_policies['consensus_threshold']:
                sorted_tools = sorted(
                    filtered_tools,
                    key=lambda t: self.tool_safety_manager.safety_scores.get(t, 0),
                    reverse=True
                )
                while len(sorted_tools) > 1:
                    avg_safety = sum(
                        self.tool_safety_manager.safety_scores.get(t, 0.5)
                        for t in sorted_tools
                    ) / len(sorted_tools)
                    if avg_safety >= self.governance_policies['consensus_threshold']:
                        break
                    sorted_tools.pop()
                filtered_tools = sorted_tools
        return filtered_tools
    
    def _select_best_tools(self, tools: List[str], max_count: int) -> List[str]:
        scored_tools = [
            (t, self.tool_safety_manager.safety_scores.get(t, 0.5))
            for t in tools
        ]
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored_tools[:max_count]]
    
    def _identify_high_risk_tools(self, tools: List[str]) -> List[str]:
        high_risk = []
        threshold = self.governance_policies['require_approval_above_risk']
        for tool in tools:
            if tool in self.tool_safety_manager.contracts:
                contract = self.tool_safety_manager.contracts[tool]
                if contract.risk_score > threshold:
                    high_risk.append(tool)
        return high_risk
    
    def _track_tool_failure(self, tool_name: str):
        failure_threshold = self.governance_policies['auto_quarantine_on_failures']
        with self.lock:
            recent_failures = sum(
                1 for record in self.governance_history
                if record.get('tool') == tool_name and
                record.get('event') == 'execution_failure' and
                record.get('timestamp', 0) > time.time() - 3600
            )
        if recent_failures >= failure_threshold:
            self.quarantine_tool(
                tool_name,
                f"Auto-quarantined after {recent_failures} failures"
            )
    
    def get_governance_stats(self) -> Dict[str, Any]:
        if self._shutdown:
            return {'status': 'shutdown'}
        with self.lock:
            recent_history = list(self.governance_history)
            total_decisions = len(recent_history)
            total_vetos = sum(
                len(h.get('vetoed_tools', [])) 
                for h in recent_history
            )
            approval_rate = (
                self.governance_metrics['approved_requests'] / 
                max(1, self.governance_metrics['total_requests'])
            )
            return {
                'total_decisions': total_decisions,
                'total_vetos': total_vetos,
                'veto_rate': total_vetos / max(1, total_decisions),
                'approval_rate': approval_rate,
                'emergency_stop_active': self.emergency_stop,
                'emergency_stop_reason': self.emergency_stop_reason,
                'quarantined_tools': list(self.quarantine_list),
                'metrics': self.governance_metrics,
                'tool_safety_stats': self.tool_safety_manager.get_global_safety_stats()
            }
    
    def update_governance_policy(self, policy_name: str, value: Any):
        if self._shutdown:
            return
        with self.lock:
            if policy_name in self.governance_policies:
                old_value = self.governance_policies[policy_name]
                self.governance_policies[policy_name] = value
                logger.info(f"Updated governance policy {policy_name}: {old_value} -> {value}")
                self.governance_history.append({
                    'timestamp': time.time(),
                    'event': 'policy_update',
                    'policy': policy_name,
                    'old_value': old_value,
                    'new_value': value
                })
            else:
                logger.warning(f"Unknown governance policy: {policy_name}")
    
    def shutdown(self):
        if self._shutdown:
            return
        
        # FIXED: Skip blocking operations during pytest runs
        is_pytest = os.environ.get("PYTEST_RUNNING") == "1"
        if is_pytest:
            self._shutdown = True
            return
        
        logging.raiseExceptions = False
        safe_log(logger.info, "Shutting down ToolSafetyGovernor...")
        self._shutdown = True
        self.tool_safety_manager.shutdown()
        with self.lock:
            threads_to_wait = list(self.quarantine_threads)
        for thread in threads_to_wait:
            if thread.is_alive():
                thread.join(timeout=1.0)
        with self.lock:
            self.governance_history.clear()
            self.quarantine_list.clear()
            self.whitelist.clear()
            self.blacklist.clear()
            self.governance_policies.clear()
            self.quarantine_threads.clear()
        safe_log(logger.info, "ToolSafetyGovernor shutdown complete")

def initialize_tool_safety():
    global _TOOL_SAFETY_INIT_DONE
    if _TOOL_SAFETY_INIT_DONE:
        logger.debug("Tool safety already initialized – skipping.")
        return ToolSafetyManager(), ToolSafetyGovernor()
    mgr = ToolSafetyManager()
    gov = ToolSafetyGovernor()
    _TOOL_SAFETY_INIT_DONE = True
    return mgr, gov